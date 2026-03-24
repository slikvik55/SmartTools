# SmartTools

Custom nodes for [ComfyUI](https://github.com/comfyanonymous/ComfyUI). Category: **slikvik** / **slikvik/Image**.

## Installation

1. Copy this folder into ComfyUI’s `custom_nodes` directory (or clone the repo there), e.g.  
   `ComfyUI/custom_nodes/SmartTools`
2. Restart ComfyUI.

Dependencies match a typical ComfyUI install: **PyTorch**, **NumPy**, **Pillow**. **Smart Image** also uses ComfyUI’s `folder_paths` and `comfy.cli_args`.

## Nodes

### Smart Resizer

**Display name:** Smart Resizer  
**Category:** `slikvik/Image`

Resizes a batch of images to a target resolution with optional **letterboxing** (pad) or **crop to fit**.

| Input | Notes |
|--------|--------|
| `image` | Image batch `(B, H, W, C)` |
| `model_type` | **VIDEO** — uses resolution preset; **IMAGE** — uses megapixel target |
| `resampling` | **Lanczos**, **Bilinear**, or **Nearest-Exact** (Pillow nearest-neighbor) |
| `Megapixels` | IMAGE mode: target size in MP (0.10–5.00) |
| `Multiple` | IMAGE mode: both sides snapped to be divisible by this (e.g. 16) |
| `VIDEO_preset` | **480p**, **720p**, or **1080p** |
| `pad_image` | On: pad (letterbox); off: crop to target aspect |

**VIDEO mode:** Picks a **square-ish** vs **wide** (16:9 / 9:16 style) target from the input aspect ratio, then applies the chosen preset dimensions.

**IMAGE mode:** Ignores `VIDEO_preset`. Chooses width/height that preserve aspect ratio, approximate total pixels `Megapixels × 1_000_000`, and satisfy `Multiple`.

**Outputs:** `image`, `width`, `height` (target dimensions).

---

### Smart Image (Smart Save)

**Display name:** Smart Image (Smart Save)  
**Category:** `slikvik`

Save / preview node with optional disk write:

- **save:** `disable` — preview only; `enable` — write to the output folder.
- **filename_prefix:** Same style as built-in save nodes (dates, tokens, etc.).
- **images:** Optional; when omitted, the node can use its cache or reload from last real saves.

Useful when you want previews to update without saving every run.

## Author

slivik (Smart Resizer header: v1.0.0 initial release).
