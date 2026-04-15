#
# SmartSave.py
#
# SaveImage-equivalent output, but writes to the output folder only when the user
# clicks "Save Image" in the UI (see web/smart_save.js + POST /smart_tools/save_image).
# Each queue run caches tensors keyed by hidden UNIQUE_ID and writes temp previews.
#

from __future__ import annotations

import json
import os
from typing import Any

import numpy as np
import torch
from PIL import Image
from PIL.PngImagePlugin import PngInfo

import folder_paths
from comfy.cli_args import args

# Last-run cache per graph node id (string). Populated in save_images; read by HTTP route.
_SMART_SAVE_CACHE: dict[str, dict[str, Any]] = {}


class SmartSave:
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        self.temp_dir = folder_paths.get_temp_directory()
        self.type = "output"
        self.prefix_append = ""
        self.compress_level = 4

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE", {"tooltip": "The images to save."}),
                "filename_prefix": (
                    "STRING",
                    {
                        "default": "ComfyUI",
                        "tooltip": (
                            "The prefix for the file to save. This may include formatting "
                            "information such as %date:yyyy-MM-dd% or %Empty Latent Image.width% "
                            "to include values from nodes."
                        ),
                    },
                ),
            },
            "hidden": {
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO",
                "unique_id": "UNIQUE_ID",
            },
        }

    RETURN_TYPES = ()
    FUNCTION = "save_images"
    OUTPUT_NODE = True
    CATEGORY = "slikvik"
    DESCRIPTION = (
        "Like Save Image, but writes to the output folder only when you click "
        "Save Image on the node (after at least one run to cache the batch)."
    )

    def _write_output_pngs(
        self,
        images: torch.Tensor,
        filename_prefix: str,
        prompt: dict | None,
        extra_pnginfo: dict | None,
    ) -> list[dict]:
        """Same file layout and metadata as official SaveImage."""
        full_output_folder, filename, counter, subfolder, filename_prefix = (
            folder_paths.get_save_image_path(
                filename_prefix,
                self.output_dir,
                images.shape[2],
                images.shape[1],
            )
        )
        results: list[dict] = []
        for batch_number, image in enumerate(images):
            i = 255.0 * image.cpu().numpy()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
            metadata = None
            if not args.disable_metadata:
                metadata = PngInfo()
                if prompt is not None:
                    metadata.add_text("prompt", json.dumps(prompt))
                if extra_pnginfo is not None:
                    for x in extra_pnginfo:
                        metadata.add_text(x, json.dumps(extra_pnginfo[x]))

            filename_with_batch_num = filename.replace("%batch_num%", str(batch_number))
            file = f"{filename_with_batch_num}_{counter:05}_.png"
            img.save(
                os.path.join(full_output_folder, file),
                pnginfo=metadata,
                compress_level=self.compress_level,
            )
            results.append(
                {
                    "filename": file,
                    "subfolder": subfolder,
                    "type": self.type,
                }
            )
            counter += 1
        return results

    def save_images(
        self,
        images: torch.Tensor,
        filename_prefix: str = "ComfyUI",
        prompt: dict | None = None,
        extra_pnginfo: dict | None = None,
        unique_id: str | int | None = None,
    ):
        filename_prefix = filename_prefix + self.prefix_append
        cache_key = str(unique_id) if unique_id is not None else None

        if cache_key is not None:
            _SMART_SAVE_CACHE[cache_key] = {
                "images": images.detach().cpu().clone(),
                "filename_prefix": filename_prefix,
                "prompt": prompt,
                "extra_pnginfo": extra_pnginfo,
            }

        # Temp previews so the node gallery updates without writing to output/.
        preview_subfolder = "SmartSavePreview"
        preview_folder = os.path.join(self.temp_dir, preview_subfolder)
        os.makedirs(preview_folder, exist_ok=True)

        results: list[dict] = []
        for batch_number, image in enumerate(images):
            arr = (255.0 * image.cpu().numpy()).clip(0, 255).astype(np.uint8)
            img = Image.fromarray(arr)
            file = f"preview_{cache_key or 'x'}_{batch_number}.png"
            full_path = os.path.join(preview_folder, file)
            img.save(full_path, compress_level=self.compress_level)
            results.append(
                {
                    "filename": file,
                    "subfolder": preview_subfolder,
                    "type": "temp",
                }
            )

        return {"ui": {"images": results}}


def smart_save_write_output(unique_id: str) -> tuple[list[dict] | None, str | None]:
    """
    Perform SaveImage-style writes from cache. Used by POST /smart_tools/save_image.
    Returns (results, None) or (None, error_message).
    """
    entry = _SMART_SAVE_CACHE.get(str(unique_id))
    if not entry:
        return None, "No cached images for this node. Run the workflow once first."

    saver = SmartSave()
    try:
        results = saver._write_output_pngs(
            entry["images"],
            entry["filename_prefix"],
            entry["prompt"],
            entry["extra_pnginfo"],
        )
    except Exception as ex:
        return None, str(ex)
    return results, None


NODE_CLASS_MAPPINGS = {
    "SmartSave": SmartSave,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SmartSave": "Smart Save",
}
