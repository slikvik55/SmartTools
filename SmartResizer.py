#
# SmartResizer.py
#
# Standalone custom node for ComfyUI.
# Resizes an image based on a preset or a target megapixel value,
# with optional padding (letterboxing) or cropping to fit.
#
# Behaviour:
# - Use Presets ON:
#     Uses 480p / 720p / 1080p / 1024px presets and simple aspect logic.
# - Use Presets OFF:
#     Uses Megapixels + Multiple (aspect-preserving target size).
# Padding, letterbox/crop, outpaint pads, feathering, and overlay mask apply in both modes.
#
# Author: slivik
# Version: 1.0.0 (Initial release)
#

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from PIL.PngImagePlugin import PngInfo

# Align with common ComfyUI image nodes for pad limits.
MAX_RESOLUTION = 16384


class SmartResizer:
    """
    A node that resizes an image to a target resolution.

    use_presets:
      - True: target size from VIDEO_preset (square-ish vs wide-ish AR).
      - False: target size from Megapixels and Multiple.

    Letterbox/crop (`pad_image`), then outpaint pads, feathering, optional mask,
    and optional overlay apply regardless of preset vs megapixel sizing.
    """

    RESOLUTIONS = ["480p", "720p", "1080p", "1024px"]
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
                "use_presets": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "label_on": "Use Presets ON",
                        "label_off": "Use Presets OFF",
                        "tooltip": (
                            "ON: target size from resolution preset (480p–1080p, 1024px). "
                            "OFF: target size from Megapixels and Multiple."
                        ),
                    },
                ),
                "resampling": (
                    cls.RESAMPLING_METHODS,
                    {"default": "Lanczos"},
                ),

                "megapixels_target": (
                    "FLOAT",
                    {
                        "default": 1.00,
                        "min": 0.10,
                        "max": 5.00,
                        "step": 0.01,
                        "display": "number",
                        "label": "Target Megapixels (MP)",
                    },
                ),
                "multiple_target": (
                    "INT",
                    {
                        "default": 16,
                        "min": 1,
                        "max": 112,
                        "step": 1,
                        "display": "number",
                        "label": "Divisible by",
                    }
                ),
                "preset_resolution": (cls.RESOLUTIONS,),
                "pad_image": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "label_on": "Pad (Letterbox)",
                        "label_off": "Crop to Fit",
                    },
                ),
                "pad_left": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": MAX_RESOLUTION,
                        "step": 8,
                        "label": "Outpaint pad left",
                    },
                ),
                "pad_top": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": MAX_RESOLUTION,
                        "step": 8,
                        "label": "Outpaint pad top",
                    },
                ),
                "pad_right": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": MAX_RESOLUTION,
                        "step": 8,
                        "label": "Outpaint pad right",
                    },
                ),
                "pad_bottom": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": MAX_RESOLUTION,
                        "step": 8,
                        "label": "Outpaint pad bottom",
                    },
                ),
                "feathering": (
                    "INT",
                    {
                        "default": 40,
                        "min": 0,
                        "max": MAX_RESOLUTION,
                        "step": 1,
                        "advanced": True,
                        "label": "Outpaint feathering",
                    },
                ),
                "overlay_mask": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "label": "Overlay Mask",
                        "label_on": "On",
                        "label_off": "Off",
                        "tooltip": (
                            "Full-opacity white where the output mask is bright (inpaint / padded "
                            "regions); unchanged where mask is 0."
                        ),
                    },
                ),
            },
            "optional": {
                "mask": (
                    "MASK",
                    {
                        "tooltip": (
                            "Optional mask for the resized frame (before outpaint pads). "
                            "Inner mask is max(your mask, edge feathering). New border stays 1."
                        ),
                    },
                ),
            },
        }

    RETURN_TYPES = ("IMAGE", "INT", "INT", "MASK")
    RETURN_NAMES = ("image", "width", "height", "mask")
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
        Compute target (width, height) when Use Presets is off:

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

    @staticmethod
    def _inner_feather_mask(d2: int, d3: int, left: int, top: int, right: int, bottom: int, feathering: int):
        t = torch.zeros((d2, d3), dtype=torch.float32)
        if feathering > 0 and feathering * 2 < d2 and feathering * 2 < d3:
            for i in range(d2):
                for j in range(d3):
                    dt = i if top != 0 else d2
                    db = d2 - i if bottom != 0 else d2
                    dl = j if left != 0 else d3
                    dr = d3 - j if right != 0 else d3
                    d_edge = min(dt, db, dl, dr)
                    if d_edge >= feathering:
                        continue
                    v = (feathering - d_edge) / feathering
                    t[i, j] = v * v
        return t

    @staticmethod
    def _prepare_source_mask(
        source_mask: torch.Tensor,
        batch: int,
        height: int,
        width: int,
        device: torch.device,
    ) -> torch.Tensor:
        """(B, height, width) float on device, clamped to [0, 1]."""
        sm = source_mask.float()
        if sm.dim() == 2:
            sm = sm.unsqueeze(0)
        if sm.shape[0] == 1 and batch > 1:
            sm = sm.expand(batch, -1, -1)
        elif sm.shape[0] != batch:
            sm = sm[0:1].expand(batch, -1, -1)
        if sm.shape[1] != height or sm.shape[2] != width:
            sm = F.interpolate(
                sm.unsqueeze(1),
                size=(height, width),
                mode="bilinear",
                align_corners=False,
            ).squeeze(1)
        return sm.clamp(0.0, 1.0).to(device)

    @classmethod
    def _outpaint_expand(
        cls,
        image: torch.Tensor,
        left: int,
        top: int,
        right: int,
        bottom: int,
        feathering: int,
        source_mask: torch.Tensor | None = None,
    ):
        """
        Canvas expansion + mask: outer pad is 1; inner is feathered edges combined with
        optional source_mask via max(source, feather) when a mask is wired.
        image: (B, H, W, C) float, any device.
        """
        image = image.to(dtype=torch.float32)
        d1, d2, d3, d4 = image.shape
        device = image.device

        new_image = (
            torch.ones(
                (d1, d2 + top + bottom, d3 + left + right, d4),
                dtype=torch.float32,
                device=device,
            )
            * 0.5
        )
        new_image[:, top : top + d2, left : left + d3, :] = image

        h_out, w_out = d2 + top + bottom, d3 + left + right
        t = cls._inner_feather_mask(d2, d3, left, top, right, bottom, feathering)
        t_b = t.to(device).unsqueeze(0).expand(d1, -1, -1)
        if source_mask is not None:
            sm = cls._prepare_source_mask(source_mask, d1, d2, d3, device)
            inner = torch.maximum(sm, t_b)
        else:
            inner = t_b

        mask = torch.ones((d1, h_out, w_out), dtype=torch.float32, device=device)
        mask[:, top : top + d2, left : left + d3] = inner

        return new_image, mask

    @staticmethod
    def _blend_mask_as_white_overlay(
        image: torch.Tensor,
        mask: torch.Tensor,
        strength: float = 1.0,
    ) -> torch.Tensor:
        """Blend RGB toward white where mask is high; image (B,H,W,C), mask (B,H,W)."""
        image = image.to(dtype=torch.float32)
        m = mask.to(dtype=torch.float32).clamp(0.0, 1.0)
        if m.dim() == 2:
            m = m.unsqueeze(0)
        b, h, w, c = image.shape
        if m.shape[0] == 1 and b > 1:
            m = m.expand(b, -1, -1)
        elif m.shape[0] != b:
            m = m[0:1].expand(b, -1, -1)
        if m.shape[1] != h or m.shape[2] != w:
            m = F.interpolate(
                m.unsqueeze(1),
                size=(h, w),
                mode="bilinear",
                align_corners=False,
            ).squeeze(1)
        m = m.to(device=image.device)
        m4 = m.unsqueeze(-1).expand(b, h, w, c)
        white = torch.ones_like(image)
        out = image * (1.0 - strength * m4) + white * (strength * m4)
        return out.clamp(0.0, 1.0)

    def process(
        self,
        image: torch.Tensor,
        use_presets: bool,
        resampling: str,
        megapixels_target: float,
        multiple_target: int,
        preset_resolution: str,
        pad_image: bool,
        pad_left: int,
        pad_top: int,
        pad_right: int,
        pad_bottom: int,
        feathering: int,
        overlay_mask: bool,
        mask: torch.Tensor | None = None,
    ):
        pil_resample = self._pil_resample(resampling)
        # Expecting shape: (batch, H, W, C)
        b, original_height, original_width, _ = image.shape

        if original_height <= 0 or original_width <= 0:
            empty_mask = torch.zeros(
                (b, original_height, original_width),
                dtype=torch.float32,
                device=image.device,
            )
            return image, original_width, original_height, empty_mask

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

        if not use_presets:
            # Megapixel target (aspect ratio + Multiple divisibility).
            if megapixels_target <= 0:
                megapixels_target = 1.0

            target_pixels = int(megapixels_target * 1_000_000)
            target_width, target_height = self._select_megapixels_target(
                original_width,
                original_height,
                target_pixels=target_pixels,
                multiple = multiple_target,
            )
        else:
            # Resolution preset behaviour.
            if preset_resolution == "480p":
                if is_square_ish:
                    target_width, target_height = 512, 512
                else:
                    if original_width < original_height:  # Portrait
                        target_width, target_height = 480, 848
                    else:  # Landscape
                        target_width, target_height = 848, 480

            elif preset_resolution == "720p":
                if is_square_ish:
                    target_width, target_height = 768, 768
                else:
                    if original_width < original_height:  # Portrait
                        target_width, target_height = 720, 1280
                    else:  # Landscape
                        target_width, target_height = 1280, 720

            elif preset_resolution == "1080p":
                if is_square_ish:
                    target_width, target_height = 1152, 1152
                else:
                    if original_width < original_height:  # Portrait
                        target_width, target_height = 1080, 1920
                    else:  # Landscape
                        target_width, target_height = 1920, 1080
            elif preset_resolution == "1024px":
                if is_square_ish:
                    target_width, target_height = 1024, 1024
                else:
                    if original_width < original_height:  # Portrait
                        target_width, target_height = 832, 1216
                    else:  # Landscape
                        target_width, target_height = 1216, 832

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

        final_batch, outpaint_mask = self._outpaint_expand(
            final_batch,
            pad_left,
            pad_top,
            pad_right,
            pad_bottom,
            feathering,
            source_mask=mask,
        )
        target_width = target_width + pad_left + pad_right
        target_height = target_height + pad_top + pad_bottom

        if overlay_mask:
            final_batch = self._blend_mask_as_white_overlay(
                final_batch, outpaint_mask
            )

        return final_batch, target_width, target_height, outpaint_mask


# --- ComfyUI Registration ---
NODE_CLASS_MAPPINGS = {
    "SmartResizerNode": SmartResizer,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "SmartResizerNode": "Smart Resizer",
}
