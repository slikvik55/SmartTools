#
# SmartLora.py
#
# Dual-model (high / low) LoRA application node, model-only.
#
# Maintains two independent, dynamically managed lists of LoRAs:
#   - "high" LoRAs are applied to the high-noise model.
#   - "low"  LoRAs are applied to the low-noise model.
#
# The whole dynamic UI lives in the frontend (web/smart_lora.js). All of
# the per-LoRA state (enabled flag, file name, strength) is funnelled into a
# single hidden STRING widget, ``lora_config``, as JSON so the backend can
# apply an arbitrary number of LoRAs per group.
#

import json
import os

import comfy.sd
import comfy.utils
import folder_paths
from aiohttp import web
from server import PromptServer


class SmartLora:
    """Applies two independent lists of model-only LoRAs to a high and a low model."""

    def __init__(self):
        self._lora_cache = {}

    @classmethod
    def INPUT_TYPES(cls):
        names = folder_paths.get_filename_list("loras")
        if not names:
            names = ["(no loras found)"]

        return {
            "optional": {
                "model_high": (
                    "MODEL",
                    {"tooltip": "High-noise diffusion model the high LoRAs are applied to."},
                ),
                "model_low": (
                    "MODEL",
                    {"tooltip": "Low-noise diffusion model the low LoRAs are applied to."},
                ),
                "prompt": (
                    "STRING",
                    {
                        "forceInput": True,
                        "tooltip": (
                            "Optional string from another node. If connected, "
                            "prepended to prompt text with a line break."
                        ),
                    },
                ),
                "prompt_text": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "",
                        "tooltip": (
                            "Optional multiline text for the prompt output; "
                            "empty if unset."
                        ),
                    },
                ),
                # Exposed only so the frontend can read the available LoRA list
                # from the node definition. Not used by the backend.
                "lora_files": (
                    names,
                    {"tooltip": "Available LoRA files (used by the node UI)."},
                ),
                # Hidden JSON state for the dynamic high/low LoRA lists.
                "lora_config": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "{}",
                        "tooltip": "Internal JSON state for the LoRA lists (managed by the UI).",
                    },
                ),
            },
        }

    RETURN_TYPES = ("MODEL", "MODEL", "STRING")
    RETURN_NAMES = ("model_high", "model_low", "prompt")
    OUTPUT_TOOLTIPS = (
        "High model with all enabled high LoRAs applied in order.",
        "Low model with all enabled low LoRAs applied in order.",
        "Combined prompt: optional prompt input (if any), line break, optional prompt text.",
    )
    FUNCTION = "apply_loras"

    CATEGORY = "slikvik"
    DISPLAY_NAME = "Smart Lora"
    DESCRIPTION = (
        "Applies two independent lists of model-only LoRAs: high LoRAs to the "
        "high model and low LoRAs to the low model. Add, remove, toggle and set "
        "the strength of each LoRA from the node UI."
    )
    SEARCH_ALIASES = [
        "lora",
        "smart lora",
        "multi lora",
        "lora stack",
        "high low lora",
        "dual lora",
    ]

    def _get_lora(self, lora_name):
        path = folder_paths.get_full_path_or_raise("loras", lora_name)
        if path not in self._lora_cache:
            try:
                self._lora_cache[path] = comfy.utils.load_torch_file(path, safe_load=True)
            except json.JSONDecodeError as e:
                try:
                    size = os.path.getsize(path)
                    with open(path, "rb") as file:
                        prefix = file.read(256).lstrip().lower()
                except OSError:
                    size = -1
                    prefix = b""

                if prefix.startswith(b"version https://git-lfs.github.com/spec"):
                    diagnosis = (
                        "The file is a Git LFS pointer rather than downloaded model weights."
                    )
                elif prefix.startswith((b"<html", b"<!doctype html")):
                    diagnosis = "The file contains an HTML download/error page instead of weights."
                elif size < 16:
                    diagnosis = "The file is empty or truncated."
                else:
                    diagnosis = "The safetensors header is corrupt or the download is incomplete."
                size_text = f"{size:,} bytes" if size >= 0 else "unknown size"
                raise RuntimeError(
                    f"SmartLora: failed to load {lora_name!r} from {path!r} ({size_text}). "
                    f"{diagnosis} Re-download or replace this LoRA file."
                ) from e
            except Exception as e:
                try:
                    size = os.path.getsize(path)
                    size_text = f"{size:,} bytes"
                except OSError:
                    size_text = "unknown size"
                raise RuntimeError(
                    f"SmartLora: failed to load {lora_name!r} from {path!r} ({size_text}). "
                    f"{type(e).__name__}: {e}"
                ) from e
        return self._lora_cache[path]

    def _merge_prompt(self, prompt_input, prompt_text):
        box = "" if prompt_text is None else str(prompt_text)
        if prompt_input is not None and str(prompt_input).strip():
            head = str(prompt_input).rstrip()
            return head + "\n" + box
        return box

    def _apply(self, model, entries):
        if model is None:
            return None

        m = model
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            if not entry.get("on", True):
                continue

            name = entry.get("name")
            if not name:
                continue

            try:
                sm_eff = float(entry.get("strength", 1.0))
            except (TypeError, ValueError):
                continue
            if sm_eff == 0:
                continue

            lora = self._get_lora(name)
            m, _ = comfy.sd.load_lora_for_models(m, None, lora, sm_eff, 0.0)

        return m

    def apply_loras(
        self,
        model_high=None,
        model_low=None,
        prompt=None,
        prompt_text=None,
        lora_files=None,
        lora_config="{}",
    ):
        try:
            cfg = json.loads(lora_config) if lora_config else {}
        except (TypeError, ValueError):
            cfg = {}
        if not isinstance(cfg, dict):
            cfg = {}

        high_entries = cfg.get("high", []) or []
        low_entries = cfg.get("low", []) or []

        out_high = self._apply(model_high, high_entries)
        out_low = self._apply(model_low, low_entries)
        out_prompt = self._merge_prompt(prompt, prompt_text)

        return (out_high, out_low, out_prompt)


@PromptServer.instance.routes.get("/smart_tools/smart_lora/lora_info")
async def smart_lora_info(request):
    """Return the sidecar JSON for a LoRA (same path, `.json` extension)."""
    name = request.query.get("name", "")
    if not name:
        return web.json_response({"found": False})

    try:
        path = folder_paths.get_full_path("loras", name)
    except Exception:
        path = None
    if not path:
        return web.json_response({"found": False})

    json_path = os.path.splitext(path)[0] + ".json"
    if not os.path.isfile(json_path):
        return web.json_response({"found": False})

    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return web.json_response({"found": True, "data": data})
    except Exception as e:
        return web.json_response({"found": False, "error": str(e)})


# ─── Saved profiles storage ──────────────────────────────────────────────────

PROFILES_FILE = os.path.join(os.path.dirname(__file__), "smart_lora_profiles.json")


def load_profiles():
    if os.path.exists(PROFILES_FILE):
        try:
            with open(PROFILES_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                return data
        except Exception:
            pass
    return {}


def save_profiles(profiles):
    with open(PROFILES_FILE, "w", encoding="utf-8") as f:
        json.dump(profiles, f, indent=2)


@PromptServer.instance.routes.get("/smart_tools/smart_lora/profiles")
async def get_profiles_handler(request):
    try:
        return web.json_response(load_profiles())
    except Exception as e:
        return web.Response(status=500, text=str(e))


@PromptServer.instance.routes.post("/smart_tools/smart_lora/profiles")
async def save_profiles_handler(request):
    try:
        data = await request.json()
        profiles = data.get("profiles", {})
        if not isinstance(profiles, dict):
            profiles = {}
        save_profiles(profiles)
        return web.Response(status=200, text="Profiles saved")
    except Exception as e:
        return web.Response(status=500, text=str(e))


NODE_CLASS_MAPPINGS = {
    "SmartLora": SmartLora,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SmartLora": "Smart Lora",
}
