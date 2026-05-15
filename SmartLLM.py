#
# SmartLLM.py
#
# Local Gemma 4 (Hugging Face snapshot: safetensors + config) via Transformers.
#

from __future__ import annotations

import base64
import gc
import io
import json
import logging
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch

logger = logging.getLogger(__name__)

_CACHE: dict[tuple[str, str], dict[str, Any]] = {}


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


def _pil_to_data_uri(pil_image: Any) -> str:
    buf = io.BytesIO()
    pil_image.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{b64}"


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


def _build_messages(
    system_prompt: str,
    user_prompt: str,
    image_urls: list[str],
) -> list[dict[str, Any]]:
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
    for url in image_urls:
        user_content.append({"type": "image", "url": url})
    user_content.append({"type": "text", "text": user_prompt or ""})
    messages.append({"role": "user", "content": user_content})
    return messages


class SmartLLM:
    """Run Gemma 4 from a local Hugging Face model directory (safetensors layout)."""

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
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "run"
    OUTPUT_NODE = False
    CATEGORY = "slikvik/LLM"
    DISPLAY_NAME = "Smart LLM"
    DESCRIPTION = (
        "Gemma 4 via Hugging Face Transformers from a local snapshot folder (safetensors). "
        "Optional `image` / `image_2` enable multimodal (vision) mode."
    )

    @classmethod
    def IS_CHANGED(
        cls,
        model_folder: str,
        system_prompt: str,
        prompt: str,
        attn_implementation: str,
        unload_model: bool,
        image: torch.Tensor | None = None,
        image_2: torch.Tensor | None = None,
    ):
        return (
            model_folder,
            system_prompt,
            prompt,
            attn_implementation,
            unload_model,
            _image_tensor_cache_key(image),
            _image_tensor_cache_key(image_2),
        )

    def run(
        self,
        model_folder: str,
        system_prompt: str,
        prompt: str,
        attn_implementation: str,
        unload_model: bool,
        image: torch.Tensor | None = None,
        image_2: torch.Tensor | None = None,
    ):
        resolved = _validate_model_folder(model_folder)
        attn_norm = _normalize_attn(attn_implementation)
        cache_key = (resolved, attn_norm)

        image_urls: list[str] = []
        if image is not None:
            image_urls.append(_pil_to_data_uri(_comfy_image_to_pil(image)))
        if image_2 is not None:
            image_urls.append(_pil_to_data_uri(_comfy_image_to_pil(image_2)))

        messages = _build_messages(system_prompt, prompt, image_urls)

        model, processor = _load_model(resolved, attn_norm)

        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            add_generation_prompt=True,
        )
        device = next(model.parameters()).device
        inputs = _move_batch_to_device(inputs, device)

        input_len = int(inputs["input_ids"].shape[1])

        gen_kw: dict[str, Any] = {**inputs, "max_new_tokens": 512}
        with torch.inference_mode():
            out_ids = model.generate(**gen_kw)

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
