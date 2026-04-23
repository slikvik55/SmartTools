#
# SmartLoad.py
#
# Load Image equivalent with optional crop rectangle (interactive UI in smart_load.js).
#

import torch

from nodes import LoadImage


class SmartLoad(LoadImage):
    """Same as Load Image; outputs are cropped to crop_x/y/w/h (image pixel space)."""

    @classmethod
    def INPUT_TYPES(cls):
        parent = LoadImage.INPUT_TYPES()
        req = dict(parent["required"])
        req["crop_x"] = ("INT", {"default": 0, "min": 0, "max": 16384, "step": 1})
        req["crop_y"] = ("INT", {"default": 0, "min": 0, "max": 16384, "step": 1})
        req["crop_w"] = ("INT", {"default": 0, "min": 0, "max": 16384, "step": 1})
        req["crop_h"] = ("INT", {"default": 0, "min": 0, "max": 16384, "step": 1})
        out = dict(parent)
        out["required"] = req
        return out

    CATEGORY = "slikvik/Image"
    DISPLAY_NAME = "Smart Load"
    DESCRIPTION = (
        "Loads an image like Load Image; use Crop to restrict the IMAGE and MASK outputs "
        "to a rectangle."
    )

    def load_image(self, image, crop_x, crop_y, crop_w, crop_h):
        img, mask = super().load_image(image)
        _, H0, W0, _ = img.shape

        if crop_w <= 0 or crop_h <= 0:
            return (img, mask)

        x, y, w, h = self._clamp_crop(crop_x, crop_y, crop_w, crop_h, W0, H0)

        img = img[:, y : y + h, x : x + w, :]

        if mask.shape[1] == H0 and mask.shape[2] == W0:
            mask = mask[:, y : y + h, x : x + w]
        else:
            b = mask.shape[0]
            mask = torch.zeros(
                (b, h, w),
                dtype=mask.dtype,
                device=mask.device,
            )

        return (img, mask)

    @staticmethod
    def _clamp_crop(x, y, w, h, W, H):
        """Match CropImage-style bounds: minimum 1 px, stay inside image."""
        min_size = 1
        xi = max(0, min(int(x), W - min_size))
        yi = max(0, min(int(y), H - min_size))
        wi = max(min_size, min(int(w), W - xi))
        hi = max(min_size, min(int(h), H - yi))
        return xi, yi, wi, hi

    @classmethod
    def IS_CHANGED(cls, image, crop_x, crop_y, crop_w, crop_h):
        return (super().IS_CHANGED(image), crop_x, crop_y, crop_w, crop_h)


NODE_CLASS_MAPPINGS = {
    "SmartLoad": SmartLoad,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SmartLoad": "Smart Load",
}
