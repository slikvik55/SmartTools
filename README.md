# SmartTools

Custom nodes for [ComfyUI](https://github.com/comfyanonymous/ComfyUI). Category: **slikvik** / **slikvik/Image**.

## Installation

1. Copy this folder into ComfyUI’s `custom_nodes` directory (or clone the repo there), e.g.  
   `ComfyUI/custom_nodes/SmartTools`
2. Restart ComfyUI.

Dependencies match a typical ComfyUI install: **PyTorch**, **NumPy**, **Pillow**. **Smart Save** also uses ComfyUI’s `folder_paths`, `comfy.cli_args`, and the Comfy server for the save route.

## Nodes

### Smart Resizer

**Display name:** Smart Resizer  
**Category:** `slikvik/Image`

Resizes a batch of images to a target resolution with optional **letterboxing** (pad) or **crop to fit**.

| Input | Notes |
|--------|--------|
| `image` | Image batch `(B, H, W, C)` |
| `use_presets` | **ON** — target size from `VIDEO_preset`; **OFF** — from `Megapixels` + `Multiple` |
| `resampling` | **Lanczos**, **Bilinear**, or **Nearest-Exact** (Pillow nearest-neighbor) |
| `Megapixels` | When presets off: target size in MP (0.10–5.00) |
| `Multiple` | When presets off: both sides snapped to be divisible by this (e.g. 16) |
| `VIDEO_preset` | **480p**, **720p**, **1080p**, or **1024px** (when presets on) |
| `pad_image` | On: pad (letterbox); off: crop to target aspect |
| Outpaint | `pad_left` / `pad_top` / `pad_right` / `pad_bottom`, `feathering`, optional `mask`, `overlay_mask` — applied after resize in **both** preset and megapixel modes |

**Use Presets ON:** Chooses **square-ish** vs **wide** (16:9 / 9:16 style) from input aspect ratio, then applies the selected preset dimensions.

**Use Presets OFF:** Ignores `VIDEO_preset`. Chooses width/height that preserve aspect ratio, approximate total pixels `Megapixels × 1_000_000`, and satisfy `Multiple`.

**Outputs:** `image`, `width`, `height`, `mask` (includes outpaint / feathering when used).

---

### Smart Save

**Display name:** Smart Save  
**Category:** `slikvik`

Same PNG output and metadata behaviour as ComfyUI’s built-in **Save Image** (`folder_paths.get_save_image_path`, `%batch_num%`, optional workflow metadata in PNG), but files are written to the **output** folder only when you click **Save Image** on the node.

1. **Queue the workflow** at least once so the node can cache the current image batch (keyed by the graph node id).
2. The node shows **temp** previews under `temp/SmartSavePreview/` so thumbnails update without writing to the final output path.
3. Click **Save Image** to write PNGs to the output directory.

Requires the included **web** extension (`web/smart_save.js`); restart ComfyUI after installing or updating this pack.

## Author

slivik (Smart Resizer header: v1.0.0 initial release).
