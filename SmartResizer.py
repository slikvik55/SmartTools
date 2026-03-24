#
# SmartResizer.py
#
# Standalone custom node for ComfyUI.
# Resizes an image based on a preset or a target megapixel value,
# with optional padding (letterboxing) or cropping to fit.
#
# Behaviour:
# - VIDEO:
#     Uses 480p / 720p / 1080p presets and simple aspect logic
# - IMAGE:
#     Ignores resolution preset and instead:
#       * Preserves original aspect ratio
#       * Targets a given megapixel count (Megapixels)
#       * Ensures both width and height are divisible by a given number (Multiple)
#
# Author: slivik
# Version: 1.0.0 (Initial release)
#

import torch
import numpy as np
from PIL import Image
from PIL.PngImagePlugin import PngInfo


class SmartResizer:
    """
    A node that resizes an image to a target resolution.

    model_type:
      - "VIDEO":
          Uses a resolution preset ("480p", "720p", "1080p").
          Chooses a square-ish or wide-ish target based on input AR.
      - "IMAGE":
          Uses Megapixels (float in MP) instead of the preset.
          The target size:
            * is as close as possible to Megapixels * 1,000,000 pixels,
            * preserves the original aspect ratio,
            * has width and height divisible by a given number (Multiple).

    The node can either pad (letterbox) or crop to fit the target size.
    """

    RESOLUTIONS = ["480p", "720p", "1080p"]
    MODEL_TYPES = ["VIDEO", "IMAGE"]
    RESAMPLING_METHODS = ["Lanczos", "Bilinear", "Nearest-Exact"]

    @staticmethod
    def _pil_resample(resampling: str):
        return {
            "Lanczos": Image.Resampling.LANCZOS,
            "Bilinear": Image.Resampling.BILINEAR,
            "Nearest-Exact": Image.Resampling.NEAREST,
        }.get(resampling, Image.Resampling.LANCZOS)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "model_type": (cls.MODEL_TYPES, {"default": "VIDEO"}),
                "resampling": (
                    cls.RESAMPLING_METHODS,
                    {"default": "Lanczos"},
                ),

                # IMAGE-only: target megapixels (MP)
                "Megapixels": (
                    "FLOAT",
                    {
                        "default": 1.00,
                        "min": 0.10,
                        "max": 5.00,
                        "step": 0.01,
                        "display": "number",
                        "label": "IMAGE only — Target Megapixels (MP)",
                    },
                ),
                "Multiple": (
                    "INT",
                    {
                        "default": 16,
                        "min": 1,
                        "max": 112,
                        "step": 1,
                        "display": "number",
                        "label": "IMAGE only — Divisible by",
                    }
                ),
                "VIDEO_preset": (cls.RESOLUTIONS,),
                "pad_image": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "label_on": "Pad (Letterbox)",
                        "label_off": "Crop to Fit",
                    },
                ),
            }
        }

    RETURN_TYPES = ("IMAGE", "INT", "INT")
    RETURN_NAMES = ("image", "width", "height")
    FUNCTION = "process"
    CATEGORY = "slikvik/Image"

    def _select_megapixels_target(
        self,
        original_width: int,
        original_height: int,
        target_pixels: int = 1_000_000,
        multiple: int = 16,
    ):
        """
        Compute target (width, height) for IMAGE mode:

        - Preserve original aspect ratio as much as possible.
        - Approximate the desired total number of pixels (target_pixels).
        - Ensure both width and height are divisible by `multiple` (default: 16).

        Returns:
            (target_width, target_height)
        """
        # Basic fallback if input size is invalid.
        if original_width <= 0 or original_height <= 0:
            side_ideal = target_pixels ** 0.5
            side = max(multiple, int(round(side_ideal / multiple) * multiple))
            return side, side

        aspect_ratio = original_width / original_height if original_height != 0 else 1.0
        if aspect_ratio <= 0:
            aspect_ratio = 1.0

        # Ideal continuous values:
        # area = W * H = target_pixels,  W / H = aspect_ratio
        # => H = sqrt(target_pixels / aspect_ratio)
        h_ideal = (target_pixels / aspect_ratio) ** 0.5

        # Consider floor and ceil candidates for H at `multiple` granularity.
        h_down = max(multiple, int(h_ideal // multiple) * multiple)
        h_up = h_down if h_down >= h_ideal else h_down + multiple

        candidates = []

        for h in {h_down, h_up}:
            if h <= 0:
                continue

            # Snap W to nearest multiple of `multiple`.
            w_float = h * aspect_ratio
            w = max(multiple, int(round(w_float / multiple) * multiple))

            area = w * h
            diff = abs(area - target_pixels)
            candidates.append((diff, area, w, h))

        if not candidates:
            # Very defensive fallback: square-ish around target_pixels.
            side = max(multiple, int(round((target_pixels ** 0.5) / multiple) * multiple))
            return side, side

        # Sort by:
        #   1) area difference from target_pixels,
        #   2) then by area (prefer smaller if equally close)
        candidates.sort(key=lambda x: (x[0], x[1]))
        _, _, best_w, best_h = candidates[0]
        return best_w, best_h

    def process(
        self,
        image: torch.Tensor,
        model_type: str,
        resampling: str,
        Megapixels: float,
        Multiple: int,
        VIDEO_preset: str,
        pad_image: bool,
    ):
        pil_resample = self._pil_resample(resampling)
        # Expecting shape: (batch, H, W, C)
        _, original_height, original_width, _ = image.shape

        if original_height <= 0 or original_width <= 0:
            # Degenerate input: just return as-is.
            return image, original_width, original_height

        # --- 1. Determine aspect ratio characteristics (for WAN presets) ---
        aspect_ratio = original_width / original_height
        ar_square = 1.0
        ar_wide_landscape = 16 / 9
        ar_wide_portrait = 9 / 16

        diff_to_square = abs(aspect_ratio - ar_square)
        diff_to_wide = min(
            abs(aspect_ratio - ar_wide_landscape),
            abs(aspect_ratio - ar_wide_portrait),
        )
        is_square_ish = diff_to_square < diff_to_wide

        # --- 2. Decide target dimensions ---
        target_width, target_height = 0, 0

        if model_type == "IMAGE":
            # Safety clamp for MP field.
            if Megapixels <= 0:
                Megapixels = 1.0

            target_pixels = int(Megapixels * 1_000_000)
            target_width, target_height = self._select_megapixels_target(
                original_width,
                original_height,
                target_pixels=target_pixels,
                multiple = Multiple,
            )
        else:
            # VIDEO mode: legacy resolution preset behaviour.
            if VIDEO_preset == "480p":
                if is_square_ish:
                    target_width, target_height = 512, 512
                else:
                    if original_width < original_height:  # Portrait
                        target_width, target_height = 480, 848
                    else:  # Landscape
                        target_width, target_height = 848, 480

            elif VIDEO_preset == "720p":
                if is_square_ish:
                    target_width, target_height = 768, 768
                else:
                    if original_width < original_height:  # Portrait
                        target_width, target_height = 720, 1280
                    else:  # Landscape
                        target_width, target_height = 1280, 720

            elif VIDEO_preset == "1080p":
                if is_square_ish:
                    target_width, target_height = 1152, 1152
                else:
                    if original_width < original_height:  # Portrait
                        target_width, target_height = 1080, 1920
                    else:  # Landscape
                        target_width, target_height = 1920, 1080

        # --- 3. Process the batch of images ---
        processed_images = []

        for img_tensor in image:
            # Convert tensor to PIL
            pil_img = Image.fromarray(
                np.clip(255.0 * img_tensor.cpu().numpy(), 0, 255).astype(np.uint8)
            )
            img_width, img_height = pil_img.size

            if pad_image:
                # --- PADDING (letterbox) ---
                original_ar = img_width / img_height if img_height != 0 else 1.0
                target_ar = target_width / target_height if target_height != 0 else 1.0

                if original_ar > target_ar:
                    # Fit to target_width, adjust height
                    scaled_width = target_width
                    scaled_height = int(target_width / original_ar)
                else:
                    # Fit to target_height, adjust width
                    scaled_height = target_height
                    scaled_width = int(target_height * original_ar)

                resized_img = pil_img.resize(
                    (scaled_width, scaled_height), pil_resample
                )

                # Create letterbox background
                background = Image.new("RGB", (target_width, target_height), (0, 0, 0))
                paste_x = (target_width - scaled_width) // 2
                paste_y = (target_height - scaled_height) // 2
                background.paste(resized_img, (paste_x, paste_y))
                final_pil_img = background

            else:
                # --- CROPPING ---
                if target_height == 0 or target_width == 0:
                    final_pil_img = pil_img
                else:
                    target_ar = target_width / target_height
                    input_ar = img_width / img_height if img_height != 0 else 1.0

                    if input_ar > target_ar:
                        # Image is wider than target: match height, crop width
                        new_height = target_height
                        new_width = int(img_width * (target_height / img_height))
                    else:
                        # Image is taller or equal AR: match width, crop height
                        new_width = target_width
                        new_height = int(img_height * (target_width / img_width))

                    resized_img = pil_img.resize(
                        (new_width, new_height), pil_resample
                    )

                    left = (new_width - target_width) / 2
                    top = (new_height - target_height) / 2
                    right = left + target_width
                    bottom = top + target_height

                    final_pil_img = resized_img.crop((left, top, right, bottom))

            # Convert back to tensor in [0, 1]
            output_np = np.array(final_pil_img).astype(np.float32) / 255.0
            output_tensor = torch.from_numpy(output_np)
            processed_images.append(output_tensor)

        final_batch = torch.stack(processed_images)

        return final_batch, target_width, target_height


# --- ComfyUI Registration ---
NODE_CLASS_MAPPINGS = {
    "SmartResizerNode": SmartResizer,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "SmartResizerNode": "Smart Resizer",
}
