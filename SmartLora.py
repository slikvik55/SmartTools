#
# SmartLora.py
#
# Load LoRA–style node with optional CLIP, up to 5 LoRAs, per-slot enable toggles.
#

import comfy.sd
import comfy.utils
import folder_paths


class SmartLora:
    """Like Load LoRA, but optional CLIP in/out, five stacked slots with enable switches."""

    def __init__(self):
        self._lora_cache = {}

    @classmethod
    def INPUT_TYPES(cls):
        names = folder_paths.get_filename_list("loras")
        if not names:
            names = ["(no loras found)"]

        req = {
            "model": (
                "MODEL",
                {"tooltip": "The diffusion model LoRA weights will be applied to."},
            ),
        }
        for i in range(1, 6):
            req[f"lora_{i}_enabled"] = (
                "BOOLEAN",
                {
                    "default": i == 1,
                    "label_on": "On",
                    "label_off": "Off",
                    "tooltip": f"Whether slot {i} is applied.",
                },
            )
            req[f"lora_{i}_name"] = (
                names,
                {"tooltip": f"LoRA file for slot {i}."},
            )
            req[f"lora_{i}_strength_model"] = (
                "FLOAT",
                {
                    "default": 1.0,
                    "min": -100.0,
                    "max": 100.0,
                    "step": 0.01,
                    "tooltip": "Strength on the diffusion model (can be negative).",
                },
            )
            req[f"lora_{i}_strength_clip"] = (
                "FLOAT",
                {
                    "default": 1.0,
                    "min": -100.0,
                    "max": 100.0,
                    "step": 0.01,
                    "tooltip": "Strength on CLIP when a CLIP input is connected.",
                },
            )

        return {
            "required": req,
            "optional": {
                "clip": (
                    "CLIP",
                    {
                        "tooltip": (
                            "Optional. Leave disconnected for model-only LoRA; "
                            "CLIP output will be None."
                        ),
                    },
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
            },
        }

    RETURN_TYPES = ("MODEL", "CLIP", "STRING")
    RETURN_NAMES = ("model", "clip", "prompt")
    OUTPUT_TOOLTIPS = (
        "Model with all enabled LoRAs applied in slot order.",
        "CLIP after enabled LoRAs, or None if CLIP was not connected.",
        "Combined prompt: optional prompt input (if any), line break, optional prompt text.",
    )
    FUNCTION = "apply_loras"

    CATEGORY = "slikvik"
    DISPLAY_NAME = "Smart Lora"
    DESCRIPTION = (
        "Applies up to five LoRAs in order (like chaining Load LoRA). "
        "CLIP input and output are optional for model-only workflows."
    )
    SEARCH_ALIASES = ["lora", "smart lora", "multi lora", "lora stack"]

    def _get_lora(self, lora_name):
        path = folder_paths.get_full_path_or_raise("loras", lora_name)
        if path not in self._lora_cache:
            self._lora_cache[path] = comfy.utils.load_torch_file(path, safe_load=True)
        return self._lora_cache[path]

    def _merge_prompt(self, prompt_input, prompt_text):
        box = "" if prompt_text is None else str(prompt_text)
        if prompt_input is not None and str(prompt_input).strip():
            head = str(prompt_input).rstrip()
            return head + "\n" + box
        return box

    def apply_loras(
        self,
        model,
        clip=None,
        prompt=None,
        prompt_text=None,
        lora_1_enabled=True,
        lora_1_name=None,
        lora_1_strength_model=1.0,
        lora_1_strength_clip=1.0,
        lora_2_enabled=False,
        lora_2_name=None,
        lora_2_strength_model=1.0,
        lora_2_strength_clip=1.0,
        lora_3_enabled=False,
        lora_3_name=None,
        lora_3_strength_model=1.0,
        lora_3_strength_clip=1.0,
        lora_4_enabled=False,
        lora_4_name=None,
        lora_4_strength_model=1.0,
        lora_4_strength_clip=1.0,
        lora_5_enabled=False,
        lora_5_name=None,
        lora_5_strength_model=1.0,
        lora_5_strength_clip=1.0,
    ):
        slots = [
            (
                lora_1_enabled,
                lora_1_name,
                lora_1_strength_model,
                lora_1_strength_clip,
            ),
            (
                lora_2_enabled,
                lora_2_name,
                lora_2_strength_model,
                lora_2_strength_clip,
            ),
            (
                lora_3_enabled,
                lora_3_name,
                lora_3_strength_model,
                lora_3_strength_clip,
            ),
            (
                lora_4_enabled,
                lora_4_name,
                lora_4_strength_model,
                lora_4_strength_clip,
            ),
            (
                lora_5_enabled,
                lora_5_name,
                lora_5_strength_model,
                lora_5_strength_clip,
            ),
        ]

        m = model
        c = clip

        for enabled, name, sm, sc in slots:
            if not enabled or not name:
                continue

            sm_eff = float(sm)
            sc_eff = 0.0 if c is None else float(sc)
            if sm_eff == 0 and sc_eff == 0:
                continue

            lora = self._get_lora(name)
            m, c = comfy.sd.load_lora_for_models(m, c, lora, sm_eff, sc_eff)

        out_prompt = self._merge_prompt(prompt, prompt_text)
        return (m, c, out_prompt)


NODE_CLASS_MAPPINGS = {
    "SmartLora": SmartLora,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SmartLora": "Smart Lora",
}
