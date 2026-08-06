# SmartTools

Custom nodes for [ComfyUI](https://github.com/comfyanonymous/ComfyUI). Category: **slikvik** / **slikvik/Image** / **slikvik/LLM**.

## Installation

1. Copy this folder into ComfyUI’s `custom_nodes` directory (or clone the repo there), e.g.  
   `ComfyUI/custom_nodes/SmartTools`
2. Restart ComfyUI.

Dependencies match a typical ComfyUI install: **PyTorch**, **NumPy**, **Pillow**. **Smart Save** also uses ComfyUI’s `folder_paths`, `comfy.cli_args`, and the Comfy server for the save route.

**Smart LLM** is optional: install Transformers-related packages from [`requirements.txt`](requirements.txt) into the same Python environment as ComfyUI (see that file for versions):
C:\APPS\AI\ComfyEasyInstall\ComfyUI-Easy-Install\python_embeded\python.exe -m pip install -r C:\APPS\AI\ComfyEasyInstall\ComfyUI-Easy-Install\ComfyUI\custom_nodes\SmartTools\requirements.txt

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

### Smart Lora

**Display name:** Smart Lora  
**Category:** `slikvik`

Applies two **independent** lists of **model-only** LoRAs in one node: **high** LoRAs to the high-noise model and **low** LoRAs to the low-noise model (e.g. for split high/low-noise model setups). No CLIP is touched.

| Input / Output | Notes |
|--------|--------|
| `model_high` (in/out) | Optional. High-noise diffusion model; **high** LoRAs are applied to it. If left unconnected, the output is `None`. |
| `model_low` (in/out) | Optional. Low-noise diffusion model; **low** LoRAs are applied to it. If left unconnected, the output is `None`. |
| `prompt` (in) | Optional string from another node; if connected it is prepended to the prompt text with a line break. |
| `prompt` (out) | Combined prompt: optional `prompt` input, line break, then `prompt_text`. |

**LoRA lists (custom UI):** Use **Add Lora (High)** / **Add Lora (Low)** to add rows to each group. Each row has:

- a **name** field (click to open a searchable LoRA picker),
- a **strength** box (click to type a value; negative values allowed),
- an **info** button (`i`) that, when a sidecar JSON exists next to the LoRA file (same name, `.json` extension), opens a modal showing the **link**, **trigger words**, and **description**, each copyable to the clipboard,
- an **enable** toggle, and
- a **delete** button (`✕`).

High LoRAs apply to `model_high` and low LoRAs to `model_low`, in list order, only when their toggle is on and strength is non-zero.

**Profiles:** Above the prompt text box, a **Profile** selector with **Save Profile As...**, **Update Profile**, and **Delete Profile** buttons lets you store and recall setups. A profile captures both LoRA lists (each LoRA's name, strength and enable toggle) plus the prompt text. Selecting a profile loads it (replacing the current lists and prompt text), **Update** overwrites the selected profile with the current state, and **Delete** removes it. Profiles are stored globally on the server in `smart_lora_profiles.json`, so they are shared across every Smart Lora node and all workflows, and persist across restarts.

**Resizing:** Drag the node wider/narrower and the rows reflow horizontally. Drag it taller/shorter and only the `prompt_text` box grows or shrinks; the LoRA rows and buttons stay fixed.

**Persistence:** The full LoRA configuration is stored as JSON on the node (in `node.properties` / the hidden `lora_config` input) and is sent to the backend, so workflows reload exactly as saved.

Requires the included **web** extension (`web/smart_lora.js`); restart ComfyUI after installing or updating this pack.

---

### Smart LLM

**Display name:** Smart LLM  
**Category:** `slikvik/LLM`

Runs **local Hugging Face vision-language** instruction checkpoints from a **model folder** (full snapshot: `config.json`, tokenizer / processor files, and `*.safetensors` or a sharded `model.safetensors.index.json`). Inference uses **Transformers** (`AutoModelForImageTextToText` + `AutoProcessor`); there is no GGUF or llama.cpp dependency.

Works with any checkpoint those Auto classes load — for example **Qwen3-VL** (including fine-tunes such as Huihui abliterated builds) when `AutoProcessor` succeeds, and **Gemma 4** (with a Gemma-specific processor fallback if AutoProcessor fails). There is no separate “model family” toggle.

| Input | Notes |
|--------|--------|
| `model_folder` | Absolute or `~` path to the directory you downloaded (e.g. with `huggingface-cli download ... --local-dir ...`). Must contain `config.json` and safetensors weights. |
| `system_prompt` | Optional multiline system message. |
| `prompt` | Multiline user prompt. |
| `max_tokens` | Cap on **new** tokens decoded after the prompt (same idea as `max_new_tokens` in Transformers). Increase if output looks truncated. |
| `attn_implementation` | Transformers attention: **`sdpa`** (default), **`eager`**, or **`flash_attention_2`** (requires `flash-attn` + CUDA; falls back to SDPA with a warning if unavailable). |
| `unload_model` | **ON** — after generation, drop the model and processor from memory and call CUDA cache cleanup so later nodes get more VRAM. **OFF** — keep the model loaded for the next run (same `model_folder` path). |
| `video_fps` | Frame rate of the optional `video` batch (default **30**). Match VideoHelperSuite `force_rate` / loaded fps so temporal grounding is correct. Ignored when `video` is disconnected. |
| `max_video_frames` | Cap on frames taken from `video` (even subsampling). Default **32**. **0** = use all frames, but batches larger than **64** are auto-capped (avoids multi‑minute hangs / VRAM blowups). Prefer VHS `frame_load_cap` for long clips. Ignored when `video` is disconnected. |
| `image` | Optional. First batch frame as RGB PIL inside the HF processor. |
| `image_2` | Optional. Second batch frame, after `image`, passed as PIL to the processor. |
| `video` | Optional. Video as an **`IMAGE` frame batch** `(B, H, W, C)` — same type as [VideoHelperSuite](https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite) Load Video **IMAGE** output. All frames are sent as native HF video (not as still images). |

**Sage Attention vs Smart LLM:** ComfyUI’s **Sage Attention** (the `sageattention` package and flags such as `--use-sage-attention`) plugs into **diffusion** sampling (`comfy` attention). **Smart LLM does not use Sage**; the VL model runs inside Hugging Face and only supports the backends above—not `sageattention`.

**VRAM:** Full VL checkpoints are large; use a size and dtype your GPU can hold, or explore quantization in Transformers separately. Video adds frame tokens — prefer VHS `frame_load_cap` and/or `max_video_frames` if generation OOMs. CUDA uses `device_map="auto"` (requires **`accelerate`**). With **Unload OFF**, the same `model_folder` + `attn_implementation` pair reuses one in-memory model (no reload each run). Changing **`model_folder`** drops any cached weights for other folders first. Changing **attention backend** replaces the previous cached copy for that folder so VRAM does not stack. (If you use several Smart LLM nodes with different folders in one workflow, the cache holds one folder at a time—whichever loads last—so the other path may reload on its next run.)

**Transformers version:** See [`requirements.txt`](requirements.txt). Gemma 4 needs **`transformers` 5.5.0 or newer**; Qwen3-VL needs a Transformers build that registers its processor/model classes.

### Smart H3 Prompt

**Display name:** Smart H3 Prompt

**Category:** `slikvik/LLM`

Uses the same local Hugging Face model cache as Smart LLM to analyze references and produce a duration-aware prompt that follows the bundled [MiniMax H3 prompt-writing guides](h3_references/). It runs a factual media-analysis pass, generates the H3 prompt deterministically, validates its structure and timing, and makes up to two text-only repair attempts if needed.

| Input | Notes |
|--------|-------|
| `skill` | **base** for T2VA/I2VA/FL2VA/L2VA, or **ref2VA** for full-reference six-section output. |
| `base_workflow` | Explicit base workflow. Ignored by ref2VA. |
| `ref_image_1_role`…`ref_image_4_role` | ref2VA only: assign each image independently as Auto, subject/reference, first frame, intermediate keyframe, last frame, or storyboard/composition. A disconnected image's role is ignored. |
| `ref_video_role` | ref2VA only: infer from the prompt, use the video for `reference generation`, treat it as a `video editing` source, or perform `video continuation`. |
| `prompt` | Creative intent and desired use of references. The base workflow does not need to be repeated here. |
| `verbatim_dialogue` | Optional dialogue/lyrics to preserve exactly inside H3 `<d>` blocks. |
| `video_duration` | Exact target duration in seconds (default **15.00**); controls final-frame alignment and valid cut range. |
| `visual_style` | **Auto** or a specific H3-compatible visual style hint. |
| `shot_count` | **0** lets the model choose; a positive value is validated as an exact shot count. |
| `audio_usage` | Infer from prompt, copy/reuse, reference only, or ignore connected audio. |
| `max_tokens` | Generation budget, default **4096** to accommodate detailed ref2VA output. |
| `image_1`…`image_4` | Optional still references. Base workflows use only their prescribed sockets; ref2VA numbers connected images densely. |
| `video` | Optional VHS-style `IMAGE` frame batch, with `video_fps` and `max_video_frames` matching Smart LLM. |
| `audio` | Optional standard `AUDIO` dictionary, directly connectable from VideoHelperSuite **Load Audio**. |

Base image mapping:

- **T2VA:** text-driven; connected still images are not used or labeled.
- **I2VA:** requires `image_1`, mapped to `<Picture 1>` at 0.00 seconds.
- **FL2VA:** requires `image_1` as Picture 1 at 0.00 and `image_2` as Picture 2 at the exact final duration.
- **L2VA:** requires `image_1`, mapped to `<Picture 1>` at the exact final duration.

ref2VA uses all connected images in socket order, skipping gaps, then exposes the video and audio as `<Video 1>` and `<Audio 1>` when active. Each connected image keeps the role selected for its original socket: if only `image_1` and `image_3` are connected, they become `<Picture 1>` and `<Picture 2>` while using `ref_image_1_role` and `ref_image_3_role`. Reusable people, objects, scenes, styles, or actions are defined separately as `<Subject N>` by the model.

`base_workflow` remains separate from ref2VA task types because base workflows are mutually exclusive, while a ref2VA summary may combine several relationships. First/intermediate/last-frame roles require `keyframe completion`; subject/reference and storyboard/composition roles require `reference generation`. These can combine with the selected video and audio tasks—for example, a character-reference image + a last-frame image + an edited source video + reference-only audio requires `[keyframe completion + reference generation + video editing + audio reference]`. Any role left on **Auto from prompt** is inferred from the user prompt and media analysis.

The audio socket accepts `{"waveform": Tensor[B,C,T], "sample_rate": int}`, the same payload returned by VideoHelperSuite `LoadAudio`. Audio is mixed to mono and resampled to 16 kHz for inference. The selected checkpoint must support every connected modality; Gemma 4 E2B, E4B, and 12B variants support native audio, while many vision-language checkpoints do not. If source-audio and target-video durations differ, the prompt is instructed to describe partial reuse, trimming, continuation, or padding rather than claiming impossible 1:1 reuse.

**Outputs:** `h3_prompt` is a clean paste-ready H3 prompt with no analysis or markdown wrapper. `analysis` contains the model's factual notes about the connected references and can be left unconnected.

Because this node normally performs two model generations (analysis and prompt writing), with up to two additional generations when repair is required, it takes longer than a single Smart LLM call. The model remains loaded between passes and follows the `unload_model` setting after completion.

## Author

slivik (Smart Resizer header: v1.0.0 initial release).
