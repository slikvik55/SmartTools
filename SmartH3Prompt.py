from __future__ import annotations

import re

from .SmartH3LLM import (
    _AUDIO_USAGES,
    _BASE_WORKFLOWS,
    _REF_IMAGE_ROLES,
    _REF_VIDEO_ROLES,
    _STYLES,
    _format_duration,
    _required_ref_task_types,
    _validate_h3_output,
)


def _text_widget(
    label: str,
    tooltip: str,
    *,
    default: str = "",
) -> tuple[str, dict[str, object]]:
    return (
        "STRING",
        {
            "default": default,
            "multiline": True,
            "label": label,
            "tooltip": tooltip,
        },
    )


def _strip_heading(text: str, heading: str) -> str:
    value = (text or "").strip()
    return re.sub(rf"^\s*{re.escape(heading)}\s*:\s*", "", value, flags=re.IGNORECASE)


def _finish_sentence(text: str) -> str:
    value = (text or "").strip()
    if not value:
        return ""
    return value if value[-1] in ".!?" else value + "."


def _active_pictures(
    skill: str,
    workflow: str,
    pictures: list[str],
) -> list[str]:
    values = [(value or "").strip() for value in pictures]
    if skill == "base":
        required = {"T2VA": 0, "I2VA": 1, "FL2VA": 2, "L2VA": 1}[workflow]
        for index in range(required):
            if not values[index]:
                raise ValueError(
                    f"SmartH3Prompt: {workflow} requires a description in <Picture {index + 1}>."
                )
        return values[:required]

    populated = [index for index, value in enumerate(values) if value]
    if not populated:
        return []
    final_index = populated[-1]
    for index in range(final_index + 1):
        if not values[index]:
            raise ValueError(
                "SmartH3Prompt: ref2VA picture descriptions must be contiguous from "
                f"<Picture 1>; <Picture {index + 1}> is empty."
            )
    return values[: final_index + 1]


def _format_timestamp(seconds: float) -> str:
    milliseconds = round(float(seconds) * 1000)
    minutes, remainder = divmod(milliseconds, 60_000)
    whole_seconds, millis = divmod(remainder, 1000)
    return f"{minutes:02d}:{whole_seconds:02d}.{millis:03d}"


def _build_shots(
    shot_1: str,
    shot_2: str,
    shot_2_start_time: float,
    shot_3: str,
    shot_3_start_time: float,
    duration: float,
) -> tuple[list[str], list[float]]:
    descriptions = [(shot_1 or "").strip(), (shot_2 or "").strip(), (shot_3 or "").strip()]
    if not descriptions[0]:
        raise ValueError("SmartH3Prompt: Shot 1 must not be empty.")
    if descriptions[2] and not descriptions[1]:
        raise ValueError("SmartH3Prompt: Shot 3 cannot be used while Shot 2 is empty.")
    if any(re.search(r"(?:^|\n)\s*\[Shot\s+\d+\]", value) for value in descriptions if value):
        raise ValueError(
            "SmartH3Prompt: do not add [Shot N] headers to shot text; the node adds them."
        )

    active = descriptions[: 3 if descriptions[2] else 2 if descriptions[1] else 1]
    cuts: list[float] = []
    if len(active) >= 2:
        cuts.append(float(shot_2_start_time))
    if len(active) == 3:
        cuts.append(float(shot_3_start_time))
    if any(value <= 0 or value >= float(duration) for value in cuts):
        raise ValueError(
            "SmartH3Prompt: every populated shot start time must be greater than zero "
            "and before video_duration."
        )
    if len(cuts) == 2 and cuts[1] <= cuts[0]:
        raise ValueError("SmartH3Prompt: Shot 3 start time must be later than Shot 2.")
    return active, cuts


def _render_shots(descriptions: list[str], cuts: list[float]) -> str:
    lines = [f"[Shot 1] {descriptions[0].strip()}"]
    for index, description in enumerate(descriptions[1:], start=2):
        timestamp = _format_timestamp(cuts[index - 2])
        lines.append(f"[Shot {index}] At {timestamp}, {description.strip()}")
    return "\n".join(lines)


def _prepend_sentence(text: str, sentence: str) -> str:
    return f"{_finish_sentence(sentence)} {text.strip()}".strip()


def _append_sentence(text: str, sentence: str) -> str:
    first = _finish_sentence(text)
    second = _finish_sentence(sentence)
    return f"{first} {second}".strip()


def _resolve_image_roles(roles: list[str]) -> list[str]:
    return [
        "Subject/reference" if role == "Auto from prompt" else role
        for role in roles
    ]


def _resolve_video_role(role: str) -> str:
    return "Reference generation" if role == "Auto from prompt" else role


def _resolve_audio_usage(usage: str) -> str:
    return "Reference only" if usage == "Auto from prompt" else usage


def _base_prompt(
    workflow: str,
    duration: float,
    visual_style: str,
    pictures: list[str],
    video_description: str,
    audio_description: str,
    audio_usage: str,
    shot_descriptions: list[str],
    cuts: list[float],
    overall_soundscape: str,
    non_diegetic_music: str,
) -> str:
    shots = list(shot_descriptions)
    if visual_style != "Auto":
        shots[0] = _prepend_sentence(shots[0], f"{visual_style} style")

    if workflow == "I2VA":
        shots[0] = _prepend_sentence(
            shots[0],
            f"The opening frame fully follows <Picture 1>, described as {pictures[0]}",
        )
    elif workflow == "FL2VA":
        shots[0] = _prepend_sentence(
            shots[0],
            f"The shot begins from <Picture 1>, described as {pictures[0]}",
        )
        shots[-1] = _append_sentence(
            shots[-1],
            f"The final frame fully reaches <Picture 2>, described as {pictures[1]}",
        )
    elif workflow == "L2VA":
        shots[-1] = _append_sentence(
            shots[-1],
            f"The final frame fully reaches <Picture 1>, described as {pictures[0]}",
        )

    if video_description.strip():
        shots[0] = _append_sentence(
            shots[0],
            "Motion, camera, and timing follow this supplied video description: "
            + video_description.strip(),
        )
    if audio_description.strip() and audio_usage != "Ignore":
        usage = _resolve_audio_usage(audio_usage).lower()
        shots[0] = _append_sentence(
            shots[0],
            f"Target audio uses the supplied audio description as {usage}: "
            + audio_description.strip(),
        )

    duration_text = _format_duration(duration)
    final_shot = len(shots)
    instruction = ""
    if workflow == "I2VA":
        instruction = (
            "For the target video, at 0.00 seconds into the target video, "
            "<Picture 1> (from [Shot 1]) is fully referenced."
        )
    elif workflow == "FL2VA":
        instruction = (
            "How the reference pictures align with the target video — Picture 1 "
            "(from Shot 1) aligns with the 0.00-second mark of the target video; "
            f"Picture 2 (from Shot {final_shot}) aligns with the "
            f"{duration_text}-second mark of the target video."
        )
    elif workflow == "L2VA":
        instruction = (
            "How the reference pictures align with the target video — <Picture 1> "
            f"(from [Shot {final_shot}]) aligns with the "
            f"{duration_text}-second mark of the target video."
        )

    soundscape = _strip_heading(overall_soundscape, "overall_soundscape")
    if not soundscape:
        if audio_description.strip() and audio_usage != "Ignore":
            soundscape = (
                "Ambient sound and physical action sounds follow the shot sequence and the "
                f"supplied audio description: {audio_description.strip()}"
            )
        else:
            soundscape = (
                "Ambient sound and physical action sounds follow the events described "
                "throughout the shot sequence."
            )
    music = _strip_heading(non_diegetic_music, "non_diegetic_music") or "N/A"
    fields = (
        "integrated_multimodal_description:\n"
        + _render_shots(shots, cuts)
        + "\n\noverall_soundscape:\n"
        + soundscape
        + "\n\nnon_diegetic_music:\n"
        + music
    )
    return f"{instruction}\n\n{fields}" if instruction else fields


def _default_definition(
    index: int,
    role: str,
    description: str,
    final_shot: int,
) -> tuple[str, str]:
    picture = f"<Picture {index}>"
    subject = f"<Subject {index}>"
    if role == "Subject/reference":
        return (
            subject,
            f"{subject} is the visible subject or visual reference from {picture}, "
            f"described as {_finish_sentence(description)}",
        )
    if role == "First frame":
        return (
            picture,
            f"{picture} is the first frame of [Shot 1], showing {_finish_sentence(description)}",
        )
    if role == "Intermediate keyframe":
        return (
            picture,
            f"{picture} is an intermediate keyframe in the target sequence, "
            f"showing {_finish_sentence(description)}",
        )
    if role == "Last frame":
        return (
            picture,
            f"{picture} is the last frame of [Shot {final_shot}], "
            f"showing {_finish_sentence(description)}",
        )
    return (
        picture,
        f"{picture} is a storyboard/composition reference for the target shots, "
        f"showing {_finish_sentence(description)}",
    )


def _build_definitions(
    custom: str,
    pictures: list[str],
    image_roles: list[str],
    video_description: str,
    video_role: str,
    audio_description: str,
    audio_usage: str,
    final_shot: int,
) -> tuple[str, list[str]]:
    body = _strip_heading(custom, "subject_definitions")
    lines = [body] if body else []
    primary_labels: list[str] = []

    for index, (description, role) in enumerate(zip(pictures, image_roles), start=1):
        label, default = _default_definition(index, role, description, final_shot)
        primary_labels.append(label)
        picture_label = f"<Picture {index}>"
        if label not in body:
            lines.append(default)
        elif picture_label not in body:
            lines.append(
                f"{picture_label} is the source image for {label}, "
                f"described as {_finish_sentence(description)}"
            )

    if video_description:
        primary_labels.append("<Video 1>")
        if "<Video 1>" not in body:
            relation = {
                "Reference generation": "a motion, camera, and temporal reference",
                "Video editing": "the source video for the target video edit",
                "Video continuation": "the source video whose ending the target video continues",
            }[video_role]
            lines.append(
                f"<Video 1> is {relation}, described as "
                f"{_finish_sentence(video_description)}"
            )

    if audio_description and audio_usage != "Ignore":
        primary_labels.append("<Audio 1>")
        if "<Audio 1>" not in body:
            relation = (
                "an audio signal reused in full or in part"
                if audio_usage == "Copy/reuse"
                else "an audio reference whose audible characteristics guide the target"
            )
            lines.append(
                f"<Audio 1> is {relation}, described as "
                f"{_finish_sentence(audio_description)}"
            )
    return "\n".join(line for line in lines if line.strip()), primary_labels


def _default_summary(primary_labels: list[str], final_shot: int) -> str:
    labels = ", ".join(primary_labels)
    return (
        f"The target video follows the supplied {final_shot}-shot sequence while using "
        f"{labels} according to their defined roles."
    )


def _build_retention(
    custom: str,
    pictures: list[str],
    roles: list[str],
    video_description: str,
    video_role: str,
    audio_description: str,
    audio_usage: str,
    final_shot: int,
) -> str:
    body = _strip_heading(custom, "retention_analysis")
    lines = [body] if body else []
    for index, role in enumerate(roles, start=1):
        label = f"<Subject {index}>" if role == "Subject/reference" else f"<Picture {index}>"
        if label in body:
            continue
        if role == "First frame":
            location = "[Shot 1] first frame"
        elif role == "Last frame":
            location = f"[Shot {final_shot}] last frame"
        elif role == "Intermediate keyframe":
            location = "intermediate keyframe"
        elif role == "Storyboard/composition":
            location = "shot planning and composition"
        else:
            location = "appears in the target shots"
        lines.append(
            f"{label} ({location}): fully_preserved - its defined visual role and "
            "described characteristics are retained."
        )
    if video_description and "<Video 1>" not in body:
        marker = "weak_reference" if video_role == "Reference generation" else "partially_preserved"
        lines.append(
            f"<Video 1> (whole-video relationship): {marker} - its defined "
            "structural role guides the target sequence."
        )
    if audio_description and audio_usage != "Ignore" and "<Audio 1>" not in body:
        marker = "partially_copy" if audio_usage == "Copy/reuse" else "reference"
        lines.append(
            f"<Audio 1>: {marker} - the target uses the audio according to its defined role."
        )
    return "\n".join(line for line in lines if line.strip())


def _add_ref_cues(
    shots: list[str],
    pictures: list[str],
    roles: list[str],
    video_description: str,
    video_role: str,
    audio_description: str,
    audio_usage: str,
) -> list[str]:
    result = list(shots)
    for index, role in reversed(list(enumerate(roles, start=1))):
        label = f"<Subject {index}>" if role == "Subject/reference" else f"<Picture {index}>"
        if role == "Last frame":
            result[-1] = _append_sentence(
                result[-1], f"The shot ends exactly on {label} as the final frame"
            )
        elif role == "Intermediate keyframe":
            target = 1 if len(result) > 1 else 0
            result[target] = _prepend_sentence(
                result[target], f"This part passes through {label} as an intermediate keyframe"
            )
        elif role == "Storyboard/composition":
            result[0] = _prepend_sentence(
                result[0], f"The framing and placement follow {label}"
            )
        elif role == "First frame":
            result[0] = _prepend_sentence(
                result[0], f"The target video begins exactly from {label}"
            )
        else:
            result[0] = _prepend_sentence(
                result[0], f"{label} retains the appearance described in its definition"
            )
    if video_description:
        cue = {
            "Reference generation": "<Video 1> guides the motion, camera, and temporal structure",
            "Video editing": "The target sequence directly edits <Video 1>",
            "Video continuation": "The target sequence continues from the ending of <Video 1>",
        }[video_role]
        result[0] = _prepend_sentence(result[0], cue)
    if audio_description and audio_usage != "Ignore":
        cue = (
            "<Audio 1> is reused in full or in part across the target timeline"
            if audio_usage == "Copy/reuse"
            else "<Audio 1> guides the target audio without copying the source signal"
        )
        result[0] = _append_sentence(result[0], cue)
    return result


def _ref_prompt(
    duration: float,
    visual_style: str,
    pictures: list[str],
    image_roles: list[str],
    video_description: str,
    video_role: str,
    audio_description: str,
    audio_usage: str,
    shot_descriptions: list[str],
    cuts: list[float],
    subject_definitions: str,
    summary: str,
    retention_analysis: str,
    overall_soundscape: str,
    non_diegetic_music: str,
) -> tuple[str, tuple[str, ...]]:
    if not pictures and not video_description and not (
        audio_description and audio_usage != "Ignore"
    ):
        raise ValueError("SmartH3Prompt: ref2VA requires at least one reference description.")

    final_shot = len(shot_descriptions)
    definitions, primary_labels = _build_definitions(
        subject_definitions,
        pictures,
        image_roles,
        video_description,
        video_role,
        audio_description,
        audio_usage,
        final_shot,
    )
    tasks = _required_ref_task_types(
        image_roles,
        bool(video_description),
        bool(audio_description and audio_usage != "Ignore"),
        video_role,
        audio_usage,
    )
    if not tasks:
        tasks = ("reference generation",)

    summary_body = _strip_heading(summary, "summary")
    summary_body = re.sub(r"^\s*\[[^\]]+\]\s*", "", summary_body)
    if not summary_body:
        summary_body = _default_summary(primary_labels, final_shot)
    if video_description and video_role == "Video editing" and not summary_body.startswith(
        "The target video is an edited version of <Video 1>."
    ):
        summary_body = (
            "The target video is an edited version of <Video 1>. " + summary_body
        )

    retention = _build_retention(
        retention_analysis,
        pictures,
        image_roles,
        video_description,
        video_role,
        audio_description,
        audio_usage,
        final_shot,
    )
    shots = _add_ref_cues(
        shot_descriptions,
        pictures,
        image_roles,
        video_description,
        video_role,
        audio_description,
        audio_usage,
    )
    style_line = (
        "The target video uses a coherent visual style based on the supplied reference "
        "descriptions."
        if visual_style == "Auto"
        else f"The target video uses a {visual_style} visual style."
    )
    soundscape = _strip_heading(overall_soundscape, "overall_soundscape")
    if not soundscape:
        if audio_description and audio_usage != "Ignore":
            soundscape = (
                "Ambient and physical sounds follow the shot sequence and the defined role "
                "of <Audio 1>."
            )
        else:
            soundscape = (
                "Ambient sound and physical action sounds follow the events described "
                "throughout the shot sequence."
            )
    music = _strip_heading(non_diegetic_music, "non_diegetic_music") or "N/A"
    task_prefix = " + ".join(tasks)
    prompt = (
        f"subject_definitions:\n{definitions}\n\n"
        f"summary:\n[{task_prefix}] {summary_body.strip()}\n\n"
        f"retention_analysis:\n{retention}\n\n"
        f"detailed_description:\n{style_line}\n{_render_shots(shots, cuts)}\n\n"
        f"overall_soundscape:\n{soundscape}\n\n"
        f"non_diegetic_music:\n{music}"
    )
    return prompt, tasks


class SmartH3Prompt:
    @classmethod
    def INPUT_TYPES(cls):
        media_tooltip = (
            "Describe this future H3 reference in plain English. The node inserts the "
            "corresponding H3 label; do not include a section heading."
        )
        return {
            "required": {
                "skill": (
                    ["base", "ref2VA"],
                    {"default": "base", "tooltip": "H3 base mode or full-reference mode."},
                ),
                "base_workflow": (
                    list(_BASE_WORKFLOWS),
                    {"default": "T2VA", "tooltip": "Used only when skill is base."},
                ),
                "ref_image_1_role": (list(_REF_IMAGE_ROLES), {"default": "Auto from prompt"}),
                "ref_image_2_role": (list(_REF_IMAGE_ROLES), {"default": "Auto from prompt"}),
                "ref_image_3_role": (list(_REF_IMAGE_ROLES), {"default": "Auto from prompt"}),
                "ref_image_4_role": (list(_REF_IMAGE_ROLES), {"default": "Auto from prompt"}),
                "ref_video_role": (list(_REF_VIDEO_ROLES), {"default": "Auto from prompt"}),
                "picture_1": _text_widget("<Picture 1>", media_tooltip),
                "picture_2": _text_widget("<Picture 2>", media_tooltip),
                "picture_3": _text_widget("<Picture 3>", media_tooltip),
                "picture_4": _text_widget("<Picture 4>", media_tooltip),
                "video_1": _text_widget("<Video 1>", media_tooltip),
                "audio_1": _text_widget("<Audio 1>", media_tooltip),
                "shot_1": _text_widget(
                    "Shot 1",
                    "Required. Describe the first shot without a [Shot 1] header.",
                ),
                "shot_2": _text_widget(
                    "Shot 2 (optional)",
                    "Optional. Describe the second shot without a header or timestamp.",
                ),
                "shot_2_start_time": (
                    "FLOAT",
                    {
                        "default": 5.0,
                        "min": 0.001,
                        "max": 3599.999,
                        "step": 0.001,
                        "label": "Shot 2 Start Time",
                        "tooltip": "Seconds from the beginning; ignored when Shot 2 is empty.",
                    },
                ),
                "shot_3": _text_widget(
                    "Shot 3 (optional)",
                    "Optional. Requires Shot 2; omit the header and timestamp.",
                ),
                "shot_3_start_time": (
                    "FLOAT",
                    {
                        "default": 10.0,
                        "min": 0.001,
                        "max": 3599.999,
                        "step": 0.001,
                        "label": "Shot 3 Start Time",
                        "tooltip": "Seconds from the beginning; ignored when Shot 3 is empty.",
                    },
                ),
                "verbatim_dialogue": _text_widget(
                    "Verbatim Dialogue",
                    "Exact dialogue/lyrics to verify. Put every line in a shot inside "
                    "<d>[Language] ...</d>; this field does not choose its speaker or timing.",
                ),
                "video_duration": (
                    "FLOAT",
                    {"default": 15.0, "min": 0.01, "max": 3600.0, "step": 0.01},
                ),
                "visual_style": (list(_STYLES), {"default": "Auto"}),
                "audio_usage": (list(_AUDIO_USAGES), {"default": "Auto from prompt"}),
                "subject_definitions": _text_widget(
                    "Subject Definitions",
                    "ref2VA details only; omit the heading. Define custom <Subject N>, "
                    "<Picture N>, <Video N>, and <Audio N> lines. Missing active definitions "
                    "receive generic defaults.",
                ),
                "summary": _text_widget(
                    "Summary",
                    "ref2VA details only; omit `summary:` and the task prefix. Describe the "
                    "target and reference relationships. A generic summary is used when empty.",
                ),
                "retention_analysis": _text_widget(
                    "Retention Analysis",
                    "ref2VA details only; omit the heading. Add one detailed relationship-marker "
                    "line per label. Missing active labels receive conservative defaults.",
                ),
                "overall_soundscape": _text_widget(
                    "Overall Soundscape",
                    "Describe ambience, physical sounds, and non-verbal human sounds in 1-4 "
                    "sentences. Omit the heading. A generic default is used when empty.",
                ),
                "non_diegetic_music": _text_widget(
                    "Non-diegetic Music",
                    "Describe audience-only music by instrumentation, tempo, rhythm, and "
                    "dynamics. Omit the heading. Empty defaults to N/A.",
                ),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("h3_prompt",)
    FUNCTION = "run"
    OUTPUT_NODE = False
    CATEGORY = "slikvik/Prompt"
    DISPLAY_NAME = "Smart H3 Prompt"
    DESCRIPTION = (
        "Builds and validates a MiniMax H3 prompt deterministically from text descriptions, "
        "without loading or running an LLM."
    )

    def run(
        self,
        skill: str,
        base_workflow: str,
        ref_image_1_role: str,
        ref_image_2_role: str,
        ref_image_3_role: str,
        ref_image_4_role: str,
        ref_video_role: str,
        picture_1: str,
        picture_2: str,
        picture_3: str,
        picture_4: str,
        video_1: str,
        audio_1: str,
        shot_1: str,
        shot_2: str,
        shot_2_start_time: float,
        shot_3: str,
        shot_3_start_time: float,
        verbatim_dialogue: str,
        video_duration: float,
        visual_style: str,
        audio_usage: str,
        subject_definitions: str,
        summary: str,
        retention_analysis: str,
        overall_soundscape: str,
        non_diegetic_music: str,
    ):
        normalized_skill = "ref2VA" if skill == "ref2VA" else "base"
        workflow = base_workflow if base_workflow in _BASE_WORKFLOWS else "T2VA"
        _format_duration(video_duration)
        pictures = _active_pictures(
            normalized_skill,
            workflow,
            [picture_1, picture_2, picture_3, picture_4],
        )
        shots, cuts = _build_shots(
            shot_1,
            shot_2,
            shot_2_start_time,
            shot_3,
            shot_3_start_time,
            video_duration,
        )
        video_description = (video_1 or "").strip()
        audio_description = (audio_1 or "").strip()

        required_tasks: tuple[str, ...] = ()
        roles: list[str] = []
        expect_video = False
        expect_audio = False
        if normalized_skill == "base":
            prompt = _base_prompt(
                workflow,
                video_duration,
                visual_style,
                pictures,
                video_description,
                audio_description,
                audio_usage,
                shots,
                cuts,
                overall_soundscape,
                non_diegetic_music,
            )
        else:
            roles = _resolve_image_roles(
                [
                    ref_image_1_role,
                    ref_image_2_role,
                    ref_image_3_role,
                    ref_image_4_role,
                ][: len(pictures)]
            )
            resolved_video_role = _resolve_video_role(ref_video_role)
            resolved_audio_usage = _resolve_audio_usage(audio_usage)
            expect_video = bool(video_description)
            expect_audio = bool(audio_description and resolved_audio_usage != "Ignore")
            prompt, required_tasks = _ref_prompt(
                video_duration,
                visual_style,
                pictures,
                roles,
                video_description,
                resolved_video_role,
                audio_description,
                resolved_audio_usage,
                shots,
                cuts,
                subject_definitions,
                summary,
                retention_analysis,
                overall_soundscape,
                non_diegetic_music,
            )

        errors = _validate_h3_output(
            prompt,
            normalized_skill,
            workflow,
            float(video_duration),
            verbatim_dialogue,
            len(shots),
            expected_picture_count=len(pictures),
            expect_video=expect_video,
            expect_audio=expect_audio,
            required_ref_tasks=required_tasks,
            expected_picture_roles=roles,
        )
        if errors:
            raise ValueError(
                "SmartH3Prompt: the supplied text could not form a valid H3 prompt:\n- "
                + "\n- ".join(errors)
            )
        return (prompt,)


NODE_CLASS_MAPPINGS = {"SmartH3Prompt": SmartH3Prompt}
NODE_DISPLAY_NAME_MAPPINGS = {"SmartH3Prompt": "Smart H3 Prompt"}
