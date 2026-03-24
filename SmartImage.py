import os
import json
import numpy as np
import torch
from PIL import Image
from PIL.PngImagePlugin import PngInfo

import folder_paths
from comfy.cli_args import args


class SmartImage:
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        self.temp_dir = folder_paths.get_temp_directory()
        self.type = "output"
        self.prefix_append = ""
        self.compress_level = 4

        # In-node caches
        self.cached_images = None          # last IMAGE batch
        self.cached_full_paths = []        # absolute paths of last REAL saved files

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "filename_prefix": ("STRING", {
                    "default": "ComfyUI",
                    "tooltip": (
                        "Prefix for the saved files. Supports formatting like %date:yyyy-MM-dd% "
                        "and node value tokens."
                    ),
                }),
                "save": (["disable", "enable"], {
                    "default": "disable",
                    "tooltip": "Enable to save images to disk. Disable = preview only.",
                }),
            },
            "optional": {
                "images": ("IMAGE", {
                    "tooltip": "Optional image input. If missing, node will use its cache "
                               "or reload from real saved files.",
                }),
            },
            "hidden": {
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO",
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)

    FUNCTION = "smart_save_images"
    OUTPUT_NODE = True
    CATEGORY = "slikvik"
    DESCRIPTION = (
        "Enhanced Save Image node with conditional saving, caching, and temp-based preview "
        "support so the preview updates even when Save is disabled."
    )

    def _load_images_from_files(self, paths):
        loaded = []
        for path in paths:
            if not os.path.exists(path):
                continue
            img = Image.open(path).convert("RGB")
            arr = np.array(img).astype(np.float32) / 255.0
            loaded.append(torch.from_numpy(arr))
        if not loaded:
            return None
        return torch.stack(loaded, dim=0)  # B,H,W,C

    def smart_save_images(
        self,
        filename_prefix="ComfyUI",
        save="disable",
        prompt=None,
        extra_pnginfo=None,
        images=None,
    ):
        """
        - If images provided: use & cache them.
        - If missing: try cache, then real saved files.
        - Preview always updates (temp folder when save = disable).
        - Real saves tracked for reload ONLY when save = enable.
        """

        # === Resolve image batch ===
        images_to_use = images

        if images_to_use is None:
            if self.cached_images is not None:
                images_to_use = self.cached_images
            elif self.cached_full_paths:
                reloaded = self._load_images_from_files(self.cached_full_paths)
                if reloaded is not None:
                    images_to_use = reloaded
                    self.cached_images = reloaded

            if images_to_use is None:
                raise RuntimeError(
                    "SmartImage: No input images, no cache, and no saved files to reload."
                )
        else:
            self.cached_images = images_to_use

        filename_prefix = filename_prefix + self.prefix_append
        results = []
        new_full_paths = []

        # === REAL SAVE MODE ===
        if save == "enable":
            full_output_folder, filename, counter, subfolder, filename_prefix = (
                folder_paths.get_save_image_path(
                    filename_prefix,
                    self.output_dir,
                    images_to_use[0].shape[1],
                    images_to_use[0].shape[0],
                )
            )

            for (batch_number, image) in enumerate(images_to_use):
                arr = (255.0 * image.cpu().numpy()).clip(0, 255).astype(np.uint8)
                img = Image.fromarray(arr)

                metadata = None
                if not args.disable_metadata:
                    metadata = PngInfo()
                    if prompt is not None:
                        metadata.add_text("prompt", json.dumps(prompt))
                    if extra_pnginfo is not None:
                        for x in extra_pnginfo:
                            metadata.add_text(x, json.dumps(extra_pnginfo[x]))

                filename_with_batch_num = filename.replace(
                    "%batch_num%", str(batch_number)
                )
                file = f"{filename_with_batch_num}_{counter:05}_.png"
                full_path = os.path.join(full_output_folder, file)

                img.save(full_path, pnginfo=metadata, compress_level=self.compress_level)

                results.append({
                    "filename": file,
                    "subfolder": subfolder,
                    "type": self.type,
                })

                new_full_paths.append(full_path)
                counter += 1

            # Track ONLY real saved files
            if new_full_paths:
                self.cached_full_paths = new_full_paths

        # === PREVIEW MODE (NO REAL SAVES) ===
        else:
            preview_subfolder = "SmartImagePreview"
            preview_folder = os.path.join(self.temp_dir, preview_subfolder)
            os.makedirs(preview_folder, exist_ok=True)

            for (batch_number, image) in enumerate(images_to_use):
                arr = (255.0 * image.cpu().numpy()).clip(0, 255).astype(np.uint8)
                img = Image.fromarray(arr)

                file = f"preview_{batch_number}.png"
                full_path = os.path.join(preview_folder, file)

                img.save(full_path, compress_level=self.compress_level)

                # These files only exist for UI preview
                results.append({
                    "filename": file,
                    "subfolder": preview_subfolder,
                    "type": "temp",  # correct ComfyUI temp file server type
                })

        return {
            "ui": {"images": results},
            "result": (images_to_use,),
        }


NODE_CLASS_MAPPINGS = {
    "SmartImage": SmartImage,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SmartImage": "Smart Image (Smart Save)",
}
