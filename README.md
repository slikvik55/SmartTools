# SmartTools

Custom nodes for [ComfyUI](https://github.com/comfyanonymous/ComfyUI). Category: **slikvik** / **slikvik/Image** / **slikvik/LLM**.

## Installation

1. Copy this folder into ComfyUI’s `custom_nodes` directory (or clone the repo there), e.g.  
   `ComfyUI/custom_nodes/SmartTools`
2. Restart ComfyUI.

Dependencies match a typical ComfyUI install: **PyTorch**, **NumPy**, **Pillow**. **Smart Save** also uses ComfyUI’s `folder_paths`, `comfy.cli_args`, and the Comfy server for the save route.

**Smart LLM** is optional: install Transformers-related packages from [`requirements-llm.txt`](requirements-llm.txt) into the same Python environment as ComfyUI (see that file for versions).

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

---

### Smart LLM

**Display name:** Smart LLM  
**Category:** `slikvik/LLM`

Runs **Google Gemma 4** instruction checkpoints from a **local Hugging Face model folder** (full snapshot: `config.json`, tokenizer / processor files, and `*.safetensors` or a sharded `model.safetensors.index.json`). Inference uses **Hugging Face Transformers** (`AutoModelForImageTextToText` + `AutoProcessor`); there is no GGUF or llama.cpp dependency.

| Input | Notes |
|--------|--------|
| `model_folder` | Absolute or `~` path to the directory you downloaded (e.g. with `huggingface-cli download google/gemma-4-2b-it --local-dir ...`). Must contain `config.json` and safetensors weights. |
| `system_prompt` | Optional multiline system message. |
| `prompt` | Multiline user prompt. |
| `max_tokens` | Cap on **new** tokens decoded after the prompt (same idea as `max_new_tokens` in Transformers). Increase if output looks truncated. |
| `attn_implementation` | Transformers attention: **`sdpa`** (default), **`eager`**, or **`flash_attention_2`** (requires `flash-attn` + CUDA; falls back to SDPA with a warning if unavailable). |
| `unload_model` | **ON** — after generation, drop the model and processor from memory and call CUDA cache cleanup so later nodes get more VRAM. **OFF** — keep the model loaded for the next run (same `model_folder` path). |
| `image` | Optional. First batch frame as RGB PIL inside the HF processor (same pattern as reference VL nodes). |
| `image_2` | Optional. Second batch frame, after `image`, passed as PIL to the processor. |

**Sage Attention vs Smart LLM:** ComfyUI’s **Sage Attention** (the `sageattention` package and flags such as `--use-sage-attention`) plugs into **diffusion** sampling (`comfy` attention). **Smart LLM does not use Sage**; Gemma runs inside Hugging Face and only supports the backends above—not `sageattention`.

**VRAM:** Full Gemma 4 checkpoints are large; use a size and dtype your GPU can hold, or explore quantization in Transformers separately. CUDA uses `device_map="auto"` (requires **`accelerate`**). With **Unload OFF**, the same `model_folder` + `attn_implementation` pair reuses one in-memory model (no reload each run). Changing **`model_folder`** drops any cached weights for other folders first. Changing **attention backend** replaces the previous cached copy for that folder so VRAM does not stack. (If you use several Smart LLM nodes with different folders in one workflow, the cache holds one folder at a time—whichever loads last—so the other path may reload on its next run.)

**Transformers version:** Gemma 4 needs **`transformers` 5.5.0 or newer** (first release with Gemma 4); see [`requirements-llm.txt`](requirements-llm.txt).

## Author

slivik (Smart Resizer header: v1.0.0 initial release).
