#
# SmartLLM.py
#
# Local HF vision-language models (AutoProcessor + AutoModelForImageTextToText)
# from a safetensors snapshot folder. Qwen3-VL and Gemma 4 are supported paths.
#

from __future__ import annotations

import gc
import json
import logging
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch

logger = logging.getLogger(__name__)

_CACHE: dict[tuple[str, str], dict[str, Any]] = {}

# Soft cap when max_video_frames=0 (use-all). Full VHS batches at 30fps easily
# hang/OOM Qwen3-VL with no Comfy progress updates.
_SAFE_MAX_VIDEO_FRAMES = 64


def _require_transformers():
    try:
        from transformers import AutoModelForImageTextToText, AutoProcessor
    except ImportError as e:
        raise RuntimeError(
            "SmartLLM requires the `transformers` package (and usually `accelerate` for CUDA). "
            "Install with: pip install -r requirements-llm.txt (or pip install \"transformers>=5.5.0\" safetensors accelerate)"
        ) from e
    return AutoModelForImageTextToText, AutoProcessor


def _validate_model_folder(model_folder: str) -> str:
    root = Path(os.path.expanduser(os.path.expandvars(model_folder.strip())))
    if not root.is_dir():
        raise FileNotFoundError(
            f"SmartLLM: model_folder is not a directory: {model_folder!r} (resolved: {root})"
        )
    if not (root / "config.json").is_file():
        raise FileNotFoundError(
            f"SmartLLM: missing config.json under model_folder: {root}. "
            "Use a full Hugging Face model snapshot (config + tokenizer + safetensors)."
        )
    st = list(root.glob("*.safetensors"))
    has_index = (root / "model.safetensors.index.json").is_file()
    if not st and not has_index:
        raise FileNotFoundError(
            f"SmartLLM: no *.safetensors and no model.safetensors.index.json in {root}. "
            "Download the full checkpoint weights into this folder."
        )
    return str(root.resolve())


def _comfy_image_to_pil(image: torch.Tensor) -> Any:
    from PIL import Image

    if image.ndim != 4:
        raise ValueError(f"SmartLLM: expected IMAGE batch (B,H,W,C), got shape {tuple(image.shape)}")
    frame = image[0].detach().cpu().clamp(0.0, 1.0).numpy()
    rgb = (frame * 255.0).round().astype(np.uint8)
    return Image.fromarray(rgb, mode="RGB")


def _frame_tensor_to_pil(frame: torch.Tensor) -> Any:
    from PIL import Image

    arr = frame.detach().cpu().clamp(0.0, 1.0).numpy()
    rgb = (arr * 255.0).round().astype(np.uint8)
    if rgb.ndim == 2:
        return Image.fromarray(rgb, mode="L").convert("RGB")
    if rgb.shape[-1] == 4:
        return Image.fromarray(rgb, mode="RGBA").convert("RGB")
    return Image.fromarray(rgb, mode="RGB")


def _even_frame_indices(n: int, max_frames: int) -> list[int]:
    if max_frames <= 0 or n <= max_frames:
        return list(range(n))
    if max_frames == 1:
        return [n // 2]
    return [int(round(i * (n - 1) / (max_frames - 1))) for i in range(max_frames)]


def _resolve_video_frame_budget(total_frames: int, max_video_frames: int) -> int:
    """Return effective max frames to keep (always >= 1 when total_frames >= 1)."""
    n = int(total_frames)
    req = int(max_video_frames)
    if req > 0:
        return min(n, req)
    if n > _SAFE_MAX_VIDEO_FRAMES:
        logger.warning(
            "SmartLLM: video has %d frames and max_video_frames=0 (use all). "
            "Capping to %d evenly spaced frames to avoid long hangs / VRAM blowups. "
            "Raise max_video_frames explicitly if you really want more (also limit upstream "
            "with VHS frame_load_cap).",
            n,
            _SAFE_MAX_VIDEO_FRAMES,
        )
        return _SAFE_MAX_VIDEO_FRAMES
    return n


def _comfy_batch_to_pils(image: torch.Tensor, max_frames: int = 0) -> list[Any]:
    """Convert IMAGE batch to PIL frames, subsampling tensor indices first when capped."""
    if image.ndim != 4:
        raise ValueError(f"SmartLLM: expected IMAGE batch (B,H,W,C), got shape {tuple(image.shape)}")
    n = int(image.shape[0])
    if n < 1:
        raise ValueError("SmartLLM: video IMAGE batch is empty.")
    budget = _resolve_video_frame_budget(n, max_frames)
    idxs = _even_frame_indices(n, budget)
    _throw_if_interrupted()
    pils: list[Any] = []
    for i in idxs:
        pils.append(_frame_tensor_to_pil(image[i]))
        if len(pils) % 16 == 0:
            _throw_if_interrupted()
    if budget < n:
        logger.info("SmartLLM: using %d / %d video frames (even subsample).", budget, n)
    else:
        logger.info("SmartLLM: using all %d video frames.", n)
    return pils


def _throw_if_interrupted() -> None:
    try:
        import comfy.model_management as model_management

        model_management.throw_exception_if_processing_interrupted()
    except ImportError:
        pass


def _generation_stopping_criteria() -> list[Any] | None:
    """Stop HF generate() when ComfyUI cancel is requested."""
    try:
        from transformers import StoppingCriteria, StoppingCriteriaList
    except ImportError:
        return None

    class _ComfyInterruptCriteria(StoppingCriteria):
        def __call__(self, input_ids, scores, **kwargs):  # noqa: ARG002
            try:
                import comfy.model_management as model_management

                return bool(model_management.processing_interrupted())
            except Exception:
                return False

    return StoppingCriteriaList([_ComfyInterruptCriteria()])


def _build_video_metadata(pil_video: list[Any], fps: float) -> Any:
    """Metadata for pre-sampled Comfy/VHS frames so Qwen3-VL/Gemma can build timestamps."""
    n = len(pil_video)
    fps_f = float(fps) if fps and fps > 0 else 30.0
    width = height = None
    if n and hasattr(pil_video[0], "size"):
        width, height = pil_video[0].size
    meta = {
        "total_num_frames": n,
        "fps": fps_f,
        "duration": float(n) / fps_f,
        "frames_indices": list(range(n)),
        "width": width,
        "height": height,
        "video_backend": "smartllm_presampled",
    }
    try:
        from transformers.video_utils import VideoMetadata

        return VideoMetadata(**meta)
    except Exception:
        return meta


def _pick_dtype():
    if torch.cuda.is_available():
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16
    return torch.float32


def _move_batch_to_device(batch: Any, device: torch.device) -> Any:
    if hasattr(batch, "to"):
        return batch.to(device)
    if isinstance(batch, dict):
        out = {}
        for k, v in batch.items():
            if torch.is_tensor(v):
                out[k] = v.to(device, non_blocking=device.type == "cuda")
            else:
                out[k] = v
        return out
    return batch


def _model_type_from_config(resolved_folder: str) -> str | None:
    cfg_path = Path(resolved_folder) / "config.json"
    try:
        with open(cfg_path, encoding="utf-8") as f:
            data = json.load(f)
        mt = data.get("model_type")
        return str(mt).lower() if mt else None
    except (OSError, json.JSONDecodeError, TypeError):
        return None


def _load_gemma4_processor_composed(resolved_folder: str, local_files_only: bool) -> Any:
    """Build Gemma4Processor from sub-components when AutoProcessor cannot."""
    from transformers import AutoFeatureExtractor, AutoImageProcessor, AutoTokenizer, AutoVideoProcessor
    from transformers.processing_utils import ProcessorMixin

    try:
        from transformers.models.gemma4.processing_gemma4 import Gemma4Processor
    except ImportError as e:
        raise RuntimeError(
            "SmartLLM: this transformers build has no Gemma4Processor. "
            "Install transformers>=5.5.0 (see requirements-llm.txt)."
        ) from e

    kw = {"local_files_only": local_files_only}
    errs: list[str] = []

    def _step(name: str, fn):
        try:
            return fn()
        except Exception as ex:
            errs.append(f"{name}: {type(ex).__name__}: {ex}")
            raise

    try:
        tokenizer = _step(
            "AutoTokenizer",
            lambda: AutoTokenizer.from_pretrained(resolved_folder, padding_side="left", **kw),
        )
        image_processor = _step(
            "AutoImageProcessor",
            lambda: AutoImageProcessor.from_pretrained(resolved_folder, **kw),
        )
        feature_extractor = _step(
            "AutoFeatureExtractor",
            lambda: AutoFeatureExtractor.from_pretrained(resolved_folder, **kw),
        )
        video_processor = _step(
            "AutoVideoProcessor",
            lambda: AutoVideoProcessor.from_pretrained(resolved_folder, **kw),
        )
    except Exception as e:
        raise RuntimeError(
            "SmartLLM: could not assemble Gemma4Processor from this folder. "
            "Use a full Hugging Face snapshot (tokenizer, preprocessor_config.json, "
            "audio/video preprocessor configs as shipped with google/gemma-4-*). "
            f"Details:\n" + "\n".join(errs)
        ) from e

    proc = Gemma4Processor(
        feature_extractor=feature_extractor,
        image_processor=image_processor,
        tokenizer=tokenizer,
        video_processor=video_processor,
    )
    if not isinstance(proc, ProcessorMixin):
        raise RuntimeError("SmartLLM: internal error — composed processor is not a ProcessorMixin.")
    return proc


def _load_processor(resolved_folder: str, local_files_only: bool = True) -> Any:
    """
    Load the multimodal processor. Do not pass tokenizer-only kwargs (e.g. padding_side)
    into AutoProcessor.from_pretrained — they are forwarded to AutoImageProcessor etc.
    and cause every sub-loader to fail, yielding a misleading 'Unrecognized processing class' error.
    """
    _, AutoProcessor = _require_transformers()
    from transformers.processing_utils import ProcessorMixin

    common = {"local_files_only": local_files_only}
    try:
        proc = AutoProcessor.from_pretrained(resolved_folder, **common)
    except ValueError as e:
        if "Unrecognized processing class" not in str(e):
            raise
        proc = None
    else:
        if isinstance(proc, ProcessorMixin):
            tok = getattr(proc, "tokenizer", None)
            if tok is not None and hasattr(tok, "padding_side"):
                tok.padding_side = "left"
            return proc
        proc = None

    mt = _model_type_from_config(resolved_folder)
    if mt == "gemma4":
        try:
            return _load_gemma4_processor_composed(resolved_folder, local_files_only)
        except RuntimeError:
            raise
        except Exception:
            pass
        try:
            from transformers.models.gemma4.processing_gemma4 import Gemma4Processor

            return Gemma4Processor.from_pretrained(resolved_folder, **common)
        except Exception as e2:
            raise RuntimeError(
                "SmartLLM: AutoProcessor failed for this folder and explicit Gemma4Processor "
                "loading also failed. Ensure the directory is a complete HF model snapshot."
            ) from e2

    raise RuntimeError(
        f"SmartLLM: could not load a processor from {resolved_folder!r} (model_type={mt!r}). "
        "Install transformers>=5.5.0 with Gemma 4 support and use a full checkpoint download."
    )


def _normalize_attn(value: str) -> str:
    v = (value or "sdpa").strip().lower()
    return v if v in ("sdpa", "eager", "flash_attention_2") else "sdpa"


def _flash_attn_usable() -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        import flash_attn  # noqa: F401
    except ImportError:
        return False
    return True


def _effective_attn_implementation(normalized: str) -> str:
    if normalized == "flash_attention_2":
        if not _flash_attn_usable():
            if torch.cuda.is_available():
                logger.warning(
                    "SmartLLM: flash_attention_2 requires the flash-attn package; using sdpa instead."
                )
                return "sdpa"
            logger.warning(
                "SmartLLM: flash_attention_2 requires CUDA; using eager instead."
            )
            return "eager"
    return normalized


def _free_cache_entry(cache_key: tuple[str, str]) -> None:
    entry = _CACHE.pop(cache_key, None)
    if not entry:
        return
    model = entry.get("model")
    processor = entry.get("processor")
    del model
    del processor
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _load_model(resolved_folder: str, attn_implementation: str = "sdpa") -> tuple[Any, Any]:
    norm = _normalize_attn(attn_implementation)
    cache_key = (resolved_folder, norm)
    if cache_key in _CACHE:
        return _CACHE[cache_key]["model"], _CACHE[cache_key]["processor"]

    # Drop weights for any other model_folder so switching paths does not leave old models in VRAM.
    for k in list(_CACHE.keys()):
        if k[0] != resolved_folder:
            logger.info(
                "SmartLLM: unloading cached model for different model_folder %r (loading %r).",
                k[0],
                resolved_folder,
            )
            _free_cache_entry(k)

    # Only one cached model per folder: a different attn_implementation would otherwise
    # leave the previous model in _CACHE forever when Unload is OFF (2x+ VRAM, slower system).
    for k in list(_CACHE.keys()):
        if k[0] == resolved_folder and k != cache_key:
            logger.info(
                "SmartLLM: unloading previous cached weights for %r (cache key was %r, now %r).",
                resolved_folder,
                k[1],
                norm,
            )
            _free_cache_entry(k)

    AutoModelForImageTextToText, _ = _require_transformers()
    dtype = _pick_dtype()
    attn_eff = _effective_attn_implementation(norm)

    processor = _load_processor(resolved_folder, local_files_only=True)

    load_kw: dict[str, Any] = {
        "torch_dtype": dtype,
        "local_files_only": True,
    }
    if torch.cuda.is_available():
        load_kw["device_map"] = "auto"
        load_kw["attn_implementation"] = attn_eff
    else:
        load_kw["attn_implementation"] = attn_eff

    try:
        model = AutoModelForImageTextToText.from_pretrained(resolved_folder, **load_kw)
    except (TypeError, ValueError, ImportError, RuntimeError, OSError) as e:
        if load_kw.pop("attn_implementation", None) is None:
            raise
        logger.warning(
            "SmartLLM: attn_implementation=%r failed (%s: %s); loading without attn_implementation.",
            attn_eff,
            type(e).__name__,
            e,
        )
        model = AutoModelForImageTextToText.from_pretrained(resolved_folder, **load_kw)
    if not torch.cuda.is_available():
        model = model.to("cpu")

    model.eval()
    _CACHE[cache_key] = {"model": model, "processor": processor}
    return model, processor


def _image_tensor_cache_key(image: torch.Tensor | None) -> Any:
    if image is None:
        return None
    t = image.detach().cpu()
    return (tuple(t.shape), float(t.sum()), float(t.abs().sum()))


def _prepare_processor_inputs(
    processor: Any,
    system_prompt: str,
    user_prompt: str,
    pil_images: list[Any],
    pil_video: list[Any] | None = None,
    video_fps: float = 30.0,
) -> Any:
    """HF multimodal pattern: render chat as text, pass PIL images/videos into the processor."""
    messages: list[dict[str, Any]] = []
    sys_stripped = (system_prompt or "").strip()
    if sys_stripped:
        messages.append(
            {
                "role": "system",
                "content": [{"type": "text", "text": sys_stripped}],
            }
        )
    user_content: list[dict[str, Any]] = []
    for pil in pil_images:
        user_content.append({"type": "image", "image": pil})
    if pil_video:
        fps = float(video_fps) if video_fps and video_fps > 0 else 30.0
        # fps / sample_fps: temporal metadata for pre-sampled frame lists (Qwen3-VL / Gemma 4).
        user_content.append(
            {
                "type": "video",
                "video": pil_video,
                "fps": fps,
                "sample_fps": fps,
            }
        )
    user_content.append({"type": "text", "text": user_prompt or ""})
    messages.append({"role": "user", "content": user_content})

    chat_str = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    proc_kw: dict[str, Any] = {"text": chat_str, "return_tensors": "pt"}
    if pil_images:
        proc_kw["images"] = pil_images
    if pil_video:
        fps = float(video_fps) if video_fps and video_fps > 0 else 30.0
        metadata = _build_video_metadata(pil_video, fps)
        proc_kw["videos"] = [pil_video]
        # Qwen3-VL reads fps from video_metadata for timestamp prompts; videos_kwargs
        # alone is not enough for pre-sampled frame lists (defaults to 24 with a warning).
        proc_kw["video_metadata"] = [metadata]
        proc_kw["videos_kwargs"] = {
            "fps": fps,
            "do_sample_frames": False,
        }

    try:
        return processor(**proc_kw)
    except TypeError as e:
        msg = str(e).lower()
        if pil_video and "video_metadata" in msg:
            # Older processors: drop top-level video_metadata, keep videos_kwargs entry.
            proc_kw.pop("video_metadata", None)
            try:
                return processor(**proc_kw)
            except TypeError:
                pass
        if pil_video and ("videos" in msg or "unexpected keyword" in msg):
            raise RuntimeError(
                "SmartLLM: this model's processor rejected video inputs (`videos=`). "
                "Use a vision-language checkpoint with video support (e.g. Qwen3-VL or Gemma 4), "
                "or disconnect the `video` input."
            ) from e
        raise
    except Exception as e:
        if pil_video:
            raise RuntimeError(
                "SmartLLM: failed to process video frames with this processor. "
                "Ensure the checkpoint supports video (e.g. Qwen3-VL / Gemma 4) and that "
                f"frame count / resolution fit VRAM. Underlying error: {type(e).__name__}: {e}"
            ) from e
        raise


class SmartLLM:
    """Run a local HF VL model (AutoProcessor) from a safetensors snapshot folder."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_folder": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": False,
                        "tooltip": "Path to a full HF model folder (config.json, tokenizer, *.safetensors).",
                    },
                ),
                "system_prompt": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "tooltip": "Optional system message.",
                    },
                ),
                "prompt": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "tooltip": "User prompt / instruction.",
                    },
                ),
                "max_tokens": (
                    "INT",
                    {
                        "default": 512,
                        "min": 1,
                        "max": 8192,
                        "step": 1,
                        "tooltip": "Maximum new tokens to generate (decode budget). Raise if replies look cut off.",
                    },
                ),
                "attn_implementation": (
                    ["sdpa", "eager", "flash_attention_2"],
                    {
                        "default": "sdpa",
                        "tooltip": (
                            "Hugging Face attention backend. flash_attention_2 needs flash-attn + CUDA "
                            "(falls back if missing). Not ComfyUI Sage Attention."
                        ),
                    },
                ),
                "unload_model": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "label_on": "Unload ON",
                        "label_off": "Unload OFF",
                        "tooltip": "When ON, remove the model from VRAM after this node finishes.",
                    },
                ),
                "video_fps": (
                    "FLOAT",
                    {
                        "default": 30.0,
                        "min": 0.1,
                        "max": 120.0,
                        "step": 0.1,
                        "tooltip": (
                            "Frame rate of the `video` IMAGE batch (match VHS force_rate / loaded fps). "
                            "Used for temporal grounding; ignored when video is disconnected."
                        ),
                    },
                ),
                "max_video_frames": (
                    "INT",
                    {
                        "default": 32,
                        "min": 0,
                        "max": 4096,
                        "step": 1,
                        "tooltip": (
                            "Max frames from `video` (evenly spaced). "
                            "0 = use all frames, but batches larger than "
                            f"{_SAFE_MAX_VIDEO_FRAMES} are auto-capped for safety. "
                            "Use VHS frame_load_cap for long clips. Ignored when video is disconnected."
                        ),
                    },
                ),
            },
            "optional": {
                "image": (
                    "IMAGE",
                    {"tooltip": "Optional. First image (batch index 0) for multimodal chat."},
                ),
                "image_2": (
                    "IMAGE",
                    {"tooltip": "Optional. Second image (batch index 0), after `image` in the prompt."},
                ),
                "video": (
                    "IMAGE",
                    {
                        "tooltip": (
                            "Optional. Video as an IMAGE frame batch (e.g. VideoHelperSuite Load Video "
                            "IMAGE output). Keep max_video_frames modest (default 32); full clips at "
                            "30fps are very slow and hard to cancel mid-generate."
                        ),
                    },
                ),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "run"
    OUTPUT_NODE = False
    CATEGORY = "slikvik/LLM"
    DISPLAY_NAME = "Smart LLM"
    DESCRIPTION = (
        "Local HF vision-language model via Transformers (AutoProcessor + "
        "AutoModelForImageTextToText) from a safetensors snapshot. "
        "Optional `image` / `image_2` / `video` (IMAGE frame batch) enable multimodal mode."
    )

    @classmethod
    def IS_CHANGED(
        cls,
        model_folder: str,
        system_prompt: str,
        prompt: str,
        max_tokens: int,
        attn_implementation: str,
        unload_model: bool,
        video_fps: float = 30.0,
        max_video_frames: int = 32,
        image: torch.Tensor | None = None,
        image_2: torch.Tensor | None = None,
        video: torch.Tensor | None = None,
    ):
        return (
            model_folder,
            system_prompt,
            prompt,
            max_tokens,
            attn_implementation,
            unload_model,
            float(video_fps),
            int(max_video_frames),
            _image_tensor_cache_key(image),
            _image_tensor_cache_key(image_2),
            _image_tensor_cache_key(video),
        )

    def run(
        self,
        model_folder: str,
        system_prompt: str,
        prompt: str,
        max_tokens: int,
        attn_implementation: str,
        unload_model: bool,
        video_fps: float = 30.0,
        max_video_frames: int = 32,
        image: torch.Tensor | None = None,
        image_2: torch.Tensor | None = None,
        video: torch.Tensor | None = None,
    ):
        resolved = _validate_model_folder(model_folder)
        attn_norm = _normalize_attn(attn_implementation)
        cache_key = (resolved, attn_norm)

        _throw_if_interrupted()

        pil_images: list[Any] = []
        if image is not None:
            pil_images.append(_comfy_image_to_pil(image))
        if image_2 is not None:
            pil_images.append(_comfy_image_to_pil(image_2))

        pil_video: list[Any] | None = None
        if video is not None:
            if video.ndim == 4:
                logger.info(
                    "SmartLLM: video batch shape=%s (frames=%d). Preparing frames…",
                    tuple(video.shape),
                    int(video.shape[0]),
                )
            pil_video = _comfy_batch_to_pils(video, int(max_video_frames))

        _throw_if_interrupted()
        model, processor = _load_model(resolved, attn_norm)

        _throw_if_interrupted()
        logger.info("SmartLLM: running processor (images=%d, video_frames=%d)…", len(pil_images), len(pil_video or []))
        inputs = _prepare_processor_inputs(
            processor,
            system_prompt,
            prompt,
            pil_images,
            pil_video=pil_video,
            video_fps=float(video_fps),
        )
        device = next(model.parameters()).device
        inputs = _move_batch_to_device(inputs, device)

        input_len = int(inputs["input_ids"].shape[1])
        logger.info(
            "SmartLLM: generating (input_tokens=%d, max_new_tokens=%d)…",
            input_len,
            int(max_tokens),
        )

        gen_kw: dict[str, Any] = {**inputs, "max_new_tokens": int(max_tokens)}
        stop = _generation_stopping_criteria()
        if stop is not None:
            gen_kw["stopping_criteria"] = stop

        _throw_if_interrupted()
        with torch.inference_mode():
            out_ids = model.generate(**gen_kw)

        _throw_if_interrupted()

        new_tokens = out_ids[0, input_len:]
        ids_list = new_tokens.detach().cpu().tolist()
        del inputs
        del gen_kw
        del new_tokens
        del out_ids

        tok = getattr(processor, "tokenizer", None)
        if tok is None:
            raise RuntimeError("SmartLLM: processor has no tokenizer attribute.")
        text = tok.decode(ids_list, skip_special_tokens=True)

        if unload_model:
            _free_cache_entry(cache_key)

        return (text,)


NODE_CLASS_MAPPINGS = {
    "SmartLLM": SmartLLM,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SmartLLM": "Smart LLM",
}
