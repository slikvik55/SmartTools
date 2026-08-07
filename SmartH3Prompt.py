from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

import torch

from .SmartLLM import (
    _audio_cache_key,
    _comfy_batch_to_pils,
    _comfy_image_to_pil,
    _free_cache_entry,
    _generate_text,
    _image_tensor_cache_key,
    _load_model,
    _normalize_attn,
    _normalize_device_placement,
    _prepare_comfy_audio,
    _throw_if_interrupted,
    _validate_model_folder,
)

logger = logging.getLogger(__name__)

_REFERENCE_DIR = Path(__file__).with_name("h3_references")
_BASE_GUIDE_PATH = _REFERENCE_DIR / "base-en.txt"
_REF_GUIDE_PATH = _REFERENCE_DIR / "ref-en.txt"

_BASE_WORKFLOWS = ("T2VA", "I2VA", "FL2VA", "L2VA")
_STYLES = (
    "Auto",
    "Cinematic",
    "Live-action",
    "2D-animated",
    "3D CG",
    "Claymation",
    "Watercolor",
    "Vintage film",
)
_AUDIO_USAGES = ("Auto from prompt", "Copy/reuse", "Reference only", "Ignore")
_REF_IMAGE_ROLES = (
    "Auto from prompt",
    "Subject/reference",
    "First frame",
    "Intermediate keyframe",
    "Last frame",
    "Storyboard/composition",
)
_REF_VIDEO_ROLES = (
    "Auto from prompt",
    "Reference generation",
    "Video editing",
    "Video continuation",
)
_BASE_FIELDS = (
    "integrated_multimodal_description:",
    "overall_soundscape:",
    "non_diegetic_music:",
)
_REF_FIELDS = (
    "subject_definitions:",
    "summary:",
    "retention_analysis:",
    "detailed_description:",
    "overall_soundscape:",
    "non_diegetic_music:",
)


def _read_guide(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8").strip()
    except OSError as e:
        raise RuntimeError(f"SmartH3Prompt: required H3 guide is missing: {path}") from e


def _base_rules_and_example(workflow: str) -> tuple[str, str]:
    guide = _read_guide(_BASE_GUIDE_PATH)
    rules = guide.split("## 5. Cases", 1)[0].strip()
    case_number = _BASE_WORKFLOWS.index(workflow) + 1
    match = re.search(
        rf"### Case {case_number}:.*?(?=\n### Case \d+:|\Z)",
        guide,
        flags=re.DOTALL,
    )
    return rules, match.group(0).strip() if match else ""


def _ref_rules_and_example() -> tuple[str, str]:
    guide = _read_guide(_REF_GUIDE_PATH)
    parts = guide.split("## 7. Complete Example", 1)
    if len(parts) == 1:
        return guide, ""
    return parts[0].strip(), ("## 7. Complete Example" + parts[1]).strip()


def _format_duration(duration: float) -> str:
    value = float(duration)
    if value <= 0:
        raise ValueError("SmartH3Prompt: video_duration must be greater than zero.")
    return f"{value:.2f}"


def _active_image_tensors(
    skill: str,
    workflow: str,
    images: list[torch.Tensor | None],
) -> list[torch.Tensor]:
    if skill == "ref2VA":
        return [image for image in images if image is not None]
    required = {"T2VA": 0, "I2VA": 1, "FL2VA": 2, "L2VA": 1}[workflow]
    for index in range(required):
        if images[index] is None:
            raise ValueError(
                f"SmartH3Prompt: {workflow} requires image_{index + 1} to be connected."
            )
    return [image for image in images[:required] if image is not None]


def _active_ref_image_roles(
    images: list[torch.Tensor | None],
    roles: list[str],
) -> list[str]:
    return [role for image, role in zip(images, roles) if image is not None]


def _asset_inventory(
    skill: str,
    workflow: str,
    image_count: int,
    has_video: bool,
    has_audio: bool,
    audio_duration: float | None,
    ref_image_roles: list[str] | None = None,
    ref_video_role: str = "Auto from prompt",
) -> str:
    lines: list[str] = []
    image_roles = ref_image_roles or []
    for index in range(image_count):
        if skill == "base":
            role = {
                "I2VA": "the exact first frame at 0.00 seconds",
                "FL2VA": (
                    "the exact first frame at 0.00 seconds"
                    if index == 0
                    else "the exact last frame at the target duration"
                ),
                "L2VA": "the exact last frame at the target duration",
            }.get(workflow, "visual reference")
        else:
            selected = image_roles[index] if index < len(image_roles) else "Auto from prompt"
            role = {
                "Auto from prompt": "role inferred from the user's request",
                "Subject/reference": (
                    "subject/reference source; abstract reusable visible content into Subject labels"
                ),
                "First frame": "concrete first frame of the target video at 0.00 seconds",
                "Intermediate keyframe": (
                    "concrete intermediate keyframe at timing established by the user's request"
                ),
                "Last frame": "concrete last frame at the exact target duration",
                "Storyboard/composition": (
                    "storyboard/composition reference for shot planning, viewpoint, and placement"
                ),
            }.get(selected, "role inferred from the user's request")
        lines.append(f"- Picture {index + 1}: {role}.")
    if has_video:
        video_role = (
            "role inferred from the user's request"
            if ref_video_role == "Auto from prompt"
            else ref_video_role.lower()
        )
        lines.append(
            f"- Video 1: {video_role}; inspect its visible subjects, motion, camera, cuts, "
            "rhythm, and timing."
        )
    if has_audio:
        duration_text = f"{audio_duration:.3f} seconds" if audio_duration is not None else "unknown"
        lines.append(
            f"- Audio 1: source duration {duration_text}; inspect speech, lyrics, voice, music, "
            "ambience, rhythm, and sound effects."
        )
    return "\n".join(lines) if lines else "- No reference assets are active for this workflow."


def _analysis_system_prompt() -> str:
    return """You are the factual reference-analysis stage for a MiniMax H3 video prompt.
Analyze only the connected assets. Do not write the final H3 prompt and do not invent details.
Return concise structured English notes under exactly these headings when applicable:
PICTURES, VIDEO, AUDIO, CROSS-REFERENCE CONSISTENCY, UNCERTAINTIES.
For pictures identify visible subjects, reusable identity traits, clothing, objects, environment,
composition, lighting, style, pose, spatial relationships, and exact visible text.
For video identify subjects, actions and state changes, camera motion, cuts, pacing, and timing.
For audio identify exact intelligible speech/lyrics, language, speaker traits, music, ambience,
sound effects, beat, and uncertain spans. Mark anything unclear as [unclear]."""


def _analysis_user_prompt(inventory: str, user_prompt: str) -> str:
    return f"""ACTIVE ASSETS
{inventory}

USER'S INTENDED VIDEO
{user_prompt.strip() or "(No additional creative description supplied.)"}

Analyze the assets factually. The user's intent is context only and must not override visible or
audible evidence."""


def _workflow_instruction(workflow: str, duration: str) -> str:
    if workflow == "T2VA":
        return (
            "Use T2VA only. Start directly with integrated_multimodal_description; do not emit "
            "an image-alignment instruction or any Picture label."
        )
    if workflow == "I2VA":
        return (
            "Use I2VA only. Picture 1 is image_1 and is the exact first frame at 0.00 seconds in "
            "[Shot 1]. The first line must exactly be: For the target video, at 0.00 seconds into "
            "the target video, <Picture 1> (from [Shot 1]) is fully referenced."
        )
    if workflow == "FL2VA":
        return (
            "Use FL2VA only. Picture 1 is image_1 at 0.00 seconds and Picture 2 is image_2 at "
            f"{duration} seconds. Prefer one continuous shot unless shot_count explicitly requires "
            "more. The final shot number used in the alignment line must match the body."
        )
    return (
        "Use L2VA only. Picture 1 is image_1 and is the exact last frame at "
        f"{duration} seconds in the actual final shot. Infer a plausible preceding state and "
        "converge exactly onto it."
    )


def _controls_text(
    duration: str,
    visual_style: str,
    shot_count: int,
    audio_usage: str,
    verbatim_dialogue: str,
    audio_duration: float | None,
) -> str:
    style = "derive from references and user intent" if visual_style == "Auto" else visual_style
    shots = "Auto (choose the minimum useful number)" if int(shot_count) == 0 else str(int(shot_count))
    audio_relation = ""
    if audio_duration is not None:
        target = float(duration)
        if abs(audio_duration - target) <= 0.01:
            audio_relation = "Source audio and target duration match."
        else:
            audio_relation = (
                f"Source audio is {audio_duration:.3f}s while target video is {duration}s. "
                "Do not claim complete 1:1 reuse unless the described trim, continuation, or padding "
                "makes that relationship valid."
            )
    dialogue = verbatim_dialogue.strip() or "(none supplied)"
    return f"""TARGET CONTROLS
- Exact video duration: {duration} seconds.
- Visual style: {style}.
- Shot count: {shots}.
- Audio usage: {audio_usage}. {audio_relation}
- Verbatim dialogue/lyrics (copy every supplied line and punctuation exactly inside <d>):
{dialogue}"""


def _generation_system_prompt(skill: str, workflow: str) -> str:
    base_rules, base_example = _base_rules_and_example(workflow)
    common = """You write production-ready MiniMax H3 audiovisual prompts.
First use the supplied factual analysis, then output only the final H3 prompt: no reasoning,
markdown fences, headings outside the specification, caveats, or introductory text.
Never copy creative details from the format example. Obey exact field names and order.
All cuts must be strictly increasing and before the exact target duration. Shot numbers are
sequential, Shot 1 has no timestamp, and every later shot starts with At MM:SS.mmm.
Use natural camera-motion prose. Preserve supplied dialogue/lyrics exactly inside <d> with a
language tag. Preserve visible text exactly in double quotes. Keep dialogue and diegetic music
out of overall_soundscape. Describe non-diegetic music by instrumentation, tempo, rhythm, and
dynamics rather than abstract mood. Use N/A only where the guide permits it."""
    if skill == "base":
        return (
            f"{common}\n\nSELECTED WORKFLOW\n{workflow}\n\nOFFICIAL BASE RULES\n{base_rules}"
            f"\n\nFORMAT-ONLY EXAMPLE FOR {workflow}\n{base_example}"
        )
    ref_rules, ref_example = _ref_rules_and_example()
    return (
        f"{common}\n\nOFFICIAL SHARED BASE RULES\n{base_rules}\n\n"
        f"OFFICIAL FULL-REFERENCE RULES\n{ref_rules}\n\n"
        f"FORMAT-ONLY FULL-REFERENCE EXAMPLE\n{ref_example}"
    )


def _generation_user_prompt(
    skill: str,
    workflow: str,
    duration: str,
    inventory: str,
    analysis: str,
    user_prompt: str,
    controls: str,
    required_ref_tasks: tuple[str, ...] = (),
) -> str:
    if skill == "base":
        mode = (
            _workflow_instruction(workflow, duration)
            + " Base mode must not emit <Subject N>, <Video N>, or <Audio N> labels; incorporate "
            "useful analyzed visual and audible traits directly into the three base fields."
        )
    else:
        task_instruction = (
            "The summary task-type prefix must include: "
            + " + ".join(required_ref_tasks)
            + "."
            if required_ref_tasks
            else "Infer the summary task-type prefix from every active asset role and user intent."
        )
        mode = (
            "Use full-reference ref2VA mode. Ignore the base_workflow selector. Number connected "
            "pictures densely in the order listed, with Video 1 and Audio 1 when active. Define "
            "reusable visible content as Subject labels, and emit all six required sections. "
            + task_instruction
        )
    return f"""MODE AND MAPPING
{mode}

ACTIVE ASSET INVENTORY
{inventory}

FACTUAL REFERENCE ANALYSIS
{analysis.strip() or "(No active references; construct from text.)"}

{controls}

USER'S CREATIVE INTENT
{user_prompt.strip() or "(No additional creative description supplied.)"}

Write the final prompt now. End the described timeline exactly at {duration} seconds."""


def _sanitize_output(text: str, skill: str) -> str:
    value = (text or "").strip()
    value = re.sub(r"^\s*```(?:text|markdown)?\s*", "", value, flags=re.IGNORECASE)
    value = re.sub(r"\s*```\s*$", "", value)
    starts = ["subject_definitions:"] if skill == "ref2VA" else [
        "For the target video,",
        "How the reference pictures align with the target video",
        "integrated_multimodal_description:",
    ]
    positions = [value.find(marker) for marker in starts if value.find(marker) >= 0]
    if positions:
        value = value[min(positions):]
    if skill == "ref2VA":
        value = re.sub(
            r"<\s*(subject|picture|video|audio)\s*(\d+)\s*>",
            lambda match: f"<{match.group(1).title()} {match.group(2)}>",
            value,
            flags=re.IGNORECASE,
        )
        value = re.sub(
            r"(?<!<)\b(subject|picture|video|audio)\s+(\d+)\b(?!>)",
            lambda match: f"<{match.group(1).title()} {match.group(2)}>",
            value,
            flags=re.IGNORECASE,
        )
    return value.strip()


def _ordered_fields_errors(text: str, fields: tuple[str, ...]) -> list[str]:
    positions = [text.find(field) for field in fields]
    errors = [f"Missing required field `{field}`." for field, pos in zip(fields, positions) if pos < 0]
    present = [pos for pos in positions if pos >= 0]
    if len(present) == len(fields) and present != sorted(present):
        errors.append("Required fields are not in the prescribed order.")
    for field in fields:
        if text.count(field) > 1:
            errors.append(f"Field `{field}` appears more than once.")
    return errors


def _description_body(text: str, skill: str) -> str:
    start_field = "detailed_description:" if skill == "ref2VA" else "integrated_multimodal_description:"
    start = text.find(start_field)
    end = text.find("overall_soundscape:", start + len(start_field))
    return text[start:end] if start >= 0 and end > start else ""


def _shot_headers(body: str) -> tuple[list[int], list[tuple[int, float]]]:
    """Return actual shot headers, ignoring bracketed shot references inside prose."""
    first_position = body.find("[Shot 1]")
    if first_position < 0:
        return [], []
    cuts = [
        (int(shot), int(minutes) * 60 + float(seconds))
        for shot, minutes, seconds in re.findall(
            r"\[Shot\s+(\d+)\]\s+At\s+(\d{2}):(\d{2}\.\d{3})",
            body,
        )
    ]
    headers = [1, *(shot for shot, _ in cuts)]
    for shot in re.findall(r"(?:^|\n)\[Shot\s+(\d+)\](?!\s+At\b)", body):
        number = int(shot)
        if number != 1 and number not in headers:
            headers.append(number)
    return headers, cuts


def _ensure_picture_role_definitions(
    text: str,
    expected_picture_roles: list[str],
    shot_count: int,
) -> str:
    """Add/clarify concrete Picture-role definitions without inventing visual details."""
    if not expected_picture_roles or "subject_definitions:" not in text or "summary:" not in text:
        return text
    definitions_end = text.find("summary:")
    definitions = text[:definitions_end].rstrip()
    remainder = text[definitions_end:].lstrip()
    body = _description_body(text, "ref2VA")
    headers, _ = _shot_headers(body)
    final_shot = int(shot_count) if int(shot_count) > 0 else (max(headers) if headers else 1)
    clauses = {
        "First frame": "is the first frame of [Shot 1], serving as a concrete opening-frame anchor.",
        "Intermediate keyframe": (
            "is an intermediate keyframe whose exact placement is established in "
            "detailed_description."
        ),
        "Last frame": (
            f"is the last frame of [Shot {final_shot}], aligned with the target video's "
            "exact duration."
        ),
        "Storyboard/composition": (
            "is a storyboard/composition reference for the shot planning, viewpoint, "
            "and subject placement described below."
        ),
    }
    for index, role in enumerate(expected_picture_roles, start=1):
        clause = clauses.get(role)
        if clause is None:
            continue
        line_pattern = rf"^<Picture {index}>[^\r\n]*$"
        line_match = re.search(line_pattern, definitions, flags=re.MULTILINE)
        role_pattern = {
            "First frame": r"\b(?:first|opening) frame\b",
            "Intermediate keyframe": r"\bkeyframe\b",
            "Last frame": r"\b(?:last|final) frame\b",
            "Storyboard/composition": r"\b(?:storyboard|composition)\b",
        }[role]
        if line_match and re.search(role_pattern, line_match.group(0), flags=re.IGNORECASE):
            continue
        canonical_line = f"<Picture {index}> {clause}"
        if line_match:
            original = line_match.group(0)
            definitions = definitions.replace(
                original,
                original.rstrip().rstrip(".") + f"; it {clause}",
                1,
            )
        else:
            definitions += "\n" + canonical_line
    return definitions + "\n\n" + remainder


def _validate_h3_output(
    text: str,
    skill: str,
    workflow: str,
    duration: float,
    verbatim_dialogue: str,
    shot_count: int,
    expected_picture_count: int = 0,
    expect_video: bool = False,
    expect_audio: bool = False,
    required_ref_tasks: tuple[str, ...] = (),
    expected_picture_roles: list[str] | None = None,
) -> list[str]:
    errors = _ordered_fields_errors(text, _REF_FIELDS if skill == "ref2VA" else _BASE_FIELDS)
    duration_text = _format_duration(duration)
    body = _description_body(text, skill)
    unique_shots, cuts = _shot_headers(body)
    if not unique_shots or unique_shots != list(range(1, len(unique_shots) + 1)):
        errors.append("Shot numbers in the description must be sequential starting at Shot 1.")
    if int(shot_count) > 0 and len(unique_shots) != int(shot_count):
        errors.append(f"Expected exactly {int(shot_count)} shots, found {len(unique_shots)}.")

    cut_times = [timestamp for _, timestamp in cuts]
    if len(cuts) != max(0, len(unique_shots) - 1):
        errors.append("Every shot after Shot 1 must begin with `At MM:SS.mmm`.")
    if cut_times != sorted(cut_times) or len(set(cut_times)) != len(cut_times):
        errors.append("Cut timestamps must be strictly increasing.")
    if any(value <= 0 or value >= float(duration) for value in cut_times):
        errors.append("Every cut timestamp must be greater than zero and before the target duration.")

    if skill == "base":
        first_line = text.splitlines()[0].strip() if text else ""
        if re.search(r"<(?:Subject|Video|Audio)\s+\d+>", text):
            errors.append("Base mode must not emit full-reference Subject, Video, or Audio labels.")
        if workflow == "T2VA":
            if not first_line.startswith("integrated_multimodal_description:"):
                errors.append("T2VA must begin directly with `integrated_multimodal_description:`.")
            if re.search(r"\bPicture\s+\d+\b|<Picture\s+\d+>", text):
                errors.append("T2VA must not contain Picture references.")
        elif workflow == "I2VA":
            expected = (
                "For the target video, at 0.00 seconds into the target video, "
                "<Picture 1> (from [Shot 1]) is fully referenced."
            )
            if first_line != expected:
                errors.append("I2VA first-frame alignment instruction is not exact.")
        elif workflow == "FL2VA":
            expected_duration = f"aligns with the {duration_text}-second mark of the target video."
            if not first_line.startswith("How the reference pictures align with the target video"):
                errors.append("FL2VA must begin with the reference-picture alignment instruction.")
            if "Picture 1" not in first_line or "Picture 2" not in first_line:
                errors.append("FL2VA alignment must include Picture 1 and Picture 2.")
            if expected_duration not in first_line:
                errors.append(f"FL2VA final alignment must use exactly {duration_text} seconds.")
            final_match = re.search(r"Picture 2 \(from Shot (\d+)\)", first_line)
            if unique_shots and (
                final_match is None or int(final_match.group(1)) != unique_shots[-1]
            ):
                errors.append("FL2VA Picture 2 alignment must name the actual final shot.")
        elif workflow == "L2VA":
            if not first_line.startswith("How the reference pictures align with the target video"):
                errors.append("L2VA must begin with the final-picture alignment instruction.")
            if f"aligns with the {duration_text}-second mark of the target video." not in first_line:
                errors.append(f"L2VA final alignment must use exactly {duration_text} seconds.")
            final_match = re.search(r"<Picture 1> \(from \[Shot (\d+)\]\)", first_line)
            if unique_shots and (
                final_match is None or int(final_match.group(1)) != unique_shots[-1]
            ):
                errors.append("L2VA Picture 1 alignment must name the actual final shot.")
        if workflow in ("I2VA", "L2VA") and "<Picture 1>" not in text:
            errors.append(f"{workflow} must reference <Picture 1>.")
    else:
        definitions = text[: text.find("summary:")] if "summary:" in text else ""
        summary_match = re.search(r"summary:\s*\[([^\]]+)\]", text)
        if summary_match is None:
            errors.append("ref2VA summary must begin with a square-bracketed task-type prefix.")
        else:
            actual_tasks = {
                value.strip().lower() for value in summary_match.group(1).split("+")
            }
            for task in required_ref_tasks:
                if task.lower() not in actual_tasks:
                    errors.append(
                        f"ref2VA summary task-type prefix must include `{task}`."
                    )
            if "video editing" in required_ref_tasks:
                after_prefix = text[summary_match.end():].lstrip()
                if not after_prefix.startswith(
                    "The target video is an edited version of <Video 1>."
                ):
                    errors.append(
                        "A video-editing summary must begin with: "
                        "`The target video is an edited version of <Video 1>.`"
                    )
        picture_indices = {int(value) for value in re.findall(r"<Picture\s+(\d+)>", text)}
        if any(index < 1 or index > int(expected_picture_count) for index in picture_indices):
            errors.append("ref2VA output contains a Picture label with no connected image.")
        if re.search(r"<Video\s+(?!1>)[0-9]+>", text) or (not expect_video and "<Video 1>" in text):
            errors.append("ref2VA output contains a Video label with no matching connected video.")
        if re.search(r"<Audio\s+(?!1>)[0-9]+>", text) or (not expect_audio and "<Audio 1>" in text):
            errors.append("ref2VA output contains an Audio label with no matching connected audio.")
        for index in range(1, int(expected_picture_count) + 1):
            if f"<Picture {index}>" not in text:
                errors.append(f"Connected reference <Picture {index}> is missing from ref2VA output.")
        role_patterns = {
            "First frame": r"\b(?:first|opening) frame\b",
            "Intermediate keyframe": r"\bkeyframe\b",
            "Last frame": r"\b(?:last|final) frame\b",
            "Storyboard/composition": r"\b(?:storyboard|composition)\b",
        }
        for index, role in enumerate(expected_picture_roles or [], start=1):
            pattern = role_patterns.get(role)
            if pattern and not re.search(
                rf"^<Picture {index}>[^\r\n]*{pattern}",
                definitions,
                flags=re.IGNORECASE | re.MULTILINE,
            ):
                errors.append(
                    f"<Picture {index}> must have a standalone subject_definitions entry "
                    f"that identifies its `{role}` role."
                )
        if expect_video and "<Video 1>" not in text:
            errors.append("Connected reference <Video 1> is missing from ref2VA output.")
        if expect_audio and "<Audio 1>" not in text:
            errors.append("Connected reference <Audio 1> is missing from ref2VA output.")
        for label in sorted(set(re.findall(r"<(Subject|Picture|Video|Audio)\s+(\d+)>", text))):
            rendered = f"<{label[0]} {label[1]}>"
            if rendered not in definitions:
                errors.append(f"{rendered} is used but not defined in subject_definitions.")

    if text.count("<d>") != text.count("</d>"):
        errors.append("Dialogue/lyric <d> tags are not balanced.")
    for dialogue_block in re.findall(r"<d>(.*?)</d>", text, flags=re.DOTALL):
        if not re.match(r"\[[^\]\r\n]+\]\s+", dialogue_block):
            errors.append("Every <d> block must begin with a [Language] tag.")
    for line in (line for line in verbatim_dialogue.splitlines() if line.strip()):
        if line.strip() not in text:
            errors.append(f"Verbatim dialogue/lyric line was not preserved: {line.strip()!r}.")
    if "says in an off-screen voiceover" in text and "lips remain completely closed" not in text:
        errors.append("Voiceover requires the corresponding on-screen character's lips to remain closed.")
    return errors


def _required_ref_task_types(
    image_roles: list[str],
    has_video: bool,
    has_audio: bool,
    ref_video_role: str,
    audio_usage: str,
) -> tuple[str, ...]:
    tasks: list[str] = []
    if any(
        role in ("First frame", "Intermediate keyframe", "Last frame")
        for role in image_roles
    ):
        tasks.append("keyframe completion")
    if any(
        role in ("Subject/reference", "Storyboard/composition")
        for role in image_roles
    ):
        tasks.append("reference generation")
    if has_video and ref_video_role != "Auto from prompt":
        tasks.append(ref_video_role.lower())
    if has_audio and audio_usage == "Copy/reuse":
        tasks.append("audio reuse")
    elif has_audio and audio_usage == "Reference only":
        tasks.append("audio reference")
    return tuple(dict.fromkeys(tasks))


def _repair_prompts(
    skill: str,
    workflow: str,
    errors: list[str],
    draft: str,
    verbatim_dialogue: str,
    generation_system: str,
    generation_user: str,
) -> tuple[str, str]:
    system = generation_system + """

REPAIR INSTRUCTION
Repair a MiniMax H3 prompt using the supplied validation errors.
Return only the corrected prompt, with no markdown fence or explanation. Preserve factual content,
reference identities, all connected reference labels, their assigned roles, and verbatim dialogue.
Change only what is needed for compliance."""
    user = (
        generation_user
        + f"\n\nREPAIR TARGET\nSkill: {skill}\nWorkflow: {workflow}\nValidation errors:\n- "
        + "\n- ".join(errors)
        + f"\n\nVerbatim dialogue/lyrics:\n{verbatim_dialogue or '(none)'}"
        + f"\n\nMalformed draft:\n{draft}"
    )
    return system, user


class SmartH3Prompt:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_folder": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": False,
                        "tooltip": "Full local Hugging Face multimodal model snapshot.",
                    },
                ),
                "skill": (
                    ["base", "ref2VA"],
                    {"default": "base", "tooltip": "H3 base mode or full-reference mode."},
                ),
                "base_workflow": (
                    list(_BASE_WORKFLOWS),
                    {"default": "T2VA", "tooltip": "Used only when skill is base."},
                ),
                "ref_image_1_role": (
                    list(_REF_IMAGE_ROLES),
                    {
                        "default": "Auto from prompt",
                        "tooltip": "ref2VA only. Role of image_1; ignored when disconnected.",
                    },
                ),
                "ref_image_2_role": (
                    list(_REF_IMAGE_ROLES),
                    {
                        "default": "Auto from prompt",
                        "tooltip": "ref2VA only. Role of image_2; ignored when disconnected.",
                    },
                ),
                "ref_image_3_role": (
                    list(_REF_IMAGE_ROLES),
                    {
                        "default": "Auto from prompt",
                        "tooltip": "ref2VA only. Role of image_3; ignored when disconnected.",
                    },
                ),
                "ref_image_4_role": (
                    list(_REF_IMAGE_ROLES),
                    {
                        "default": "Auto from prompt",
                        "tooltip": "ref2VA only. Role of image_4; ignored when disconnected.",
                    },
                ),
                "ref_video_role": (
                    list(_REF_VIDEO_ROLES),
                    {
                        "default": "Auto from prompt",
                        "tooltip": "ref2VA only. How the connected source video is used.",
                    },
                ),
                "prompt": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "tooltip": "Describe the desired target video and how references should be used.",
                    },
                ),
                "verbatim_dialogue": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "tooltip": "Dialogue or lyrics to preserve exactly, one passage per line.",
                    },
                ),
                "video_duration": (
                    "FLOAT",
                    {"default": 15.0, "min": 0.01, "max": 3600.0, "step": 0.01},
                ),
                "visual_style": (list(_STYLES), {"default": "Auto"}),
                "shot_count": (
                    "INT",
                    {"default": 0, "min": 0, "max": 100, "step": 1, "tooltip": "0 = Auto."},
                ),
                "audio_usage": (list(_AUDIO_USAGES), {"default": "Auto from prompt"}),
                "max_tokens": (
                    "INT",
                    {"default": 4096, "min": 256, "max": 16384, "step": 1},
                ),
                "attn_implementation": (
                    ["sdpa", "eager", "flash_attention_2"],
                    {"default": "sdpa"},
                ),
                "unload_model": ("BOOLEAN", {"default": False}),
                "video_fps": (
                    "FLOAT",
                    {"default": 30.0, "min": 0.1, "max": 120.0, "step": 0.1},
                ),
                "max_video_frames": (
                    "INT",
                    {"default": 32, "min": 0, "max": 4096, "step": 1},
                ),
                "device_placement": (
                    ["cuda", "auto"],
                    {
                        "default": "cuda",
                        "tooltip": (
                            "cuda forces the complete HF model onto GPU after freeing Comfy's "
                            "cached models. auto permits very slow CPU offload if VRAM is tight."
                        ),
                    },
                ),
                "enable_thinking": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": (
                            "Keep OFF for H3 prompts. Reasoning-capable templates can otherwise "
                            "spend the output budget thinking and omit required H3 fields."
                        ),
                    },
                ),
            },
            "optional": {
                "image_1": ("IMAGE",),
                "image_2": ("IMAGE",),
                "image_3": ("IMAGE",),
                "image_4": ("IMAGE",),
                "video": ("IMAGE",),
                "audio": ("AUDIO",),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("h3_prompt", "analysis")
    FUNCTION = "run"
    OUTPUT_NODE = False
    CATEGORY = "slikvik/LLM"
    DISPLAY_NAME = "Smart H3 Prompt"
    DESCRIPTION = (
        "Analyzes optional picture, video, and VHS/ComfyUI audio references with a local "
        "multimodal model, then writes and validates a MiniMax H3 prompt."
    )

    @classmethod
    def IS_CHANGED(
        cls,
        model_folder: str,
        skill: str,
        base_workflow: str,
        ref_image_1_role: str,
        ref_image_2_role: str,
        ref_image_3_role: str,
        ref_image_4_role: str,
        ref_video_role: str,
        prompt: str,
        verbatim_dialogue: str,
        video_duration: float,
        visual_style: str,
        shot_count: int,
        audio_usage: str,
        max_tokens: int,
        attn_implementation: str,
        device_placement: str,
        enable_thinking: bool,
        unload_model: bool,
        video_fps: float,
        max_video_frames: int,
        image_1: torch.Tensor | None = None,
        image_2: torch.Tensor | None = None,
        image_3: torch.Tensor | None = None,
        image_4: torch.Tensor | None = None,
        video: torch.Tensor | None = None,
        audio: dict[str, Any] | None = None,
    ):
        return (
            model_folder,
            skill,
            base_workflow,
            ref_image_1_role,
            ref_image_2_role,
            ref_image_3_role,
            ref_image_4_role,
            ref_video_role,
            prompt,
            verbatim_dialogue,
            float(video_duration),
            visual_style,
            int(shot_count),
            audio_usage,
            int(max_tokens),
            attn_implementation,
            device_placement,
            bool(enable_thinking),
            bool(unload_model),
            float(video_fps),
            int(max_video_frames),
            *(_image_tensor_cache_key(image) for image in (image_1, image_2, image_3, image_4)),
            _image_tensor_cache_key(video),
            _audio_cache_key(audio),
        )

    def run(
        self,
        model_folder: str,
        skill: str,
        base_workflow: str,
        ref_image_1_role: str,
        ref_image_2_role: str,
        ref_image_3_role: str,
        ref_image_4_role: str,
        ref_video_role: str,
        prompt: str,
        verbatim_dialogue: str,
        video_duration: float,
        visual_style: str,
        shot_count: int,
        audio_usage: str,
        max_tokens: int,
        attn_implementation: str,
        device_placement: str,
        enable_thinking: bool,
        unload_model: bool,
        video_fps: float,
        max_video_frames: int,
        image_1: torch.Tensor | None = None,
        image_2: torch.Tensor | None = None,
        image_3: torch.Tensor | None = None,
        image_4: torch.Tensor | None = None,
        video: torch.Tensor | None = None,
        audio: dict[str, Any] | None = None,
    ):
        skill = "ref2VA" if skill == "ref2VA" else "base"
        workflow = base_workflow if base_workflow in _BASE_WORKFLOWS else "T2VA"
        duration_text = _format_duration(video_duration)
        image_tensors = [image_1, image_2, image_3, image_4]
        image_roles = [
            ref_image_1_role,
            ref_image_2_role,
            ref_image_3_role,
            ref_image_4_role,
        ]
        active_tensors = _active_image_tensors(
            skill, workflow, image_tensors
        )
        active_ref_roles = (
            _active_ref_image_roles(image_tensors, image_roles)
            if skill == "ref2VA"
            else []
        )
        pil_images = [_comfy_image_to_pil(image) for image in active_tensors]
        pil_video = (
            _comfy_batch_to_pils(video, int(max_video_frames)) if video is not None else None
        )
        use_audio = audio is not None and audio_usage != "Ignore"
        audio_waveform, audio_duration = _prepare_comfy_audio(audio if use_audio else None)
        inventory = _asset_inventory(
            skill,
            workflow,
            len(pil_images),
            pil_video is not None,
            audio_waveform is not None,
            audio_duration,
            ref_image_roles=active_ref_roles,
            ref_video_role=ref_video_role,
        )
        required_ref_tasks = (
            _required_ref_task_types(
                active_ref_roles,
                pil_video is not None,
                audio_waveform is not None,
                ref_video_role,
                audio_usage,
            )
            if skill == "ref2VA"
            else ()
        )
        controls = _controls_text(
            duration_text,
            visual_style,
            int(shot_count),
            audio_usage,
            verbatim_dialogue,
            audio_duration,
        )

        resolved = _validate_model_folder(model_folder)
        attn_norm = _normalize_attn(attn_implementation)
        placement = _normalize_device_placement(device_placement)
        cache_key = (resolved, attn_norm, placement)
        _throw_if_interrupted()
        model, processor = _load_model(resolved, attn_norm, placement)
        try:
            analysis = _generate_text(
                model,
                processor,
                system_prompt=_analysis_system_prompt(),
                user_prompt=_analysis_user_prompt(inventory, prompt),
                max_tokens=min(2048, max(512, int(max_tokens))),
                pil_images=pil_images,
                pil_video=pil_video,
                video_fps=float(video_fps),
                audio_waveform=audio_waveform,
                do_sample=False,
                enable_thinking=bool(enable_thinking),
            ).strip()
            generation_system = _generation_system_prompt(skill, workflow)
            generation_user = _generation_user_prompt(
                skill,
                workflow,
                duration_text,
                inventory,
                analysis,
                prompt,
                controls,
                required_ref_tasks,
            )
            draft = _generate_text(
                model,
                processor,
                system_prompt=generation_system,
                user_prompt=generation_user,
                max_tokens=int(max_tokens),
                do_sample=False,
                enable_thinking=bool(enable_thinking),
            )
            cleaned = _sanitize_output(draft, skill)
            if skill == "ref2VA":
                cleaned = _ensure_picture_role_definitions(
                    cleaned, active_ref_roles, int(shot_count)
                )
            errors = _validate_h3_output(
                cleaned,
                skill,
                workflow,
                float(video_duration),
                verbatim_dialogue,
                int(shot_count),
                expected_picture_count=len(pil_images),
                expect_video=pil_video is not None,
                expect_audio=audio_waveform is not None,
                required_ref_tasks=required_ref_tasks,
                expected_picture_roles=active_ref_roles,
            )
            max_repair_passes = 2
            for repair_attempt in range(1, max_repair_passes + 1):
                if not errors:
                    break
                logger.warning(
                    "SmartH3Prompt: repair pass %d/%d for %d validation issue(s).",
                    repair_attempt,
                    max_repair_passes,
                    len(errors),
                )
                repair_system, repair_user = _repair_prompts(
                    skill,
                    workflow,
                    errors,
                    cleaned,
                    verbatim_dialogue,
                    generation_system,
                    generation_user,
                )
                cleaned = _sanitize_output(
                    _generate_text(
                        model,
                        processor,
                        system_prompt=repair_system,
                        user_prompt=repair_user,
                        max_tokens=int(max_tokens),
                        do_sample=False,
                        enable_thinking=bool(enable_thinking),
                    ),
                    skill,
                )
                if skill == "ref2VA":
                    cleaned = _ensure_picture_role_definitions(
                        cleaned, active_ref_roles, int(shot_count)
                    )
                errors = _validate_h3_output(
                    cleaned,
                    skill,
                    workflow,
                    float(video_duration),
                    verbatim_dialogue,
                    int(shot_count),
                    expected_picture_count=len(pil_images),
                    expect_video=pil_video is not None,
                    expect_audio=audio_waveform is not None,
                    required_ref_tasks=required_ref_tasks,
                    expected_picture_roles=active_ref_roles,
                )
            if errors:
                raise RuntimeError(
                    f"SmartH3Prompt: generated prompt remained invalid after "
                    f"{max_repair_passes} repair passes:\n- "
                    + "\n- ".join(errors)
                )
            return cleaned, analysis
        finally:
            if unload_model:
                _free_cache_entry(cache_key)


NODE_CLASS_MAPPINGS = {"SmartH3Prompt": SmartH3Prompt}
NODE_DISPLAY_NAME_MAPPINGS = {"SmartH3Prompt": "Smart H3 Prompt"}
