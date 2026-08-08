from __future__ import annotations

import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest import mock

import torch


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = "_smarttools_h3_tests"
package = types.ModuleType(PACKAGE)
package.__path__ = [str(ROOT)]
sys.modules.setdefault(PACKAGE, package)


def _load(name: str):
    full_name = f"{PACKAGE}.{name}"
    if full_name in sys.modules:
        return sys.modules[full_name]
    spec = importlib.util.spec_from_file_location(full_name, ROOT / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[full_name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


llm = _load("SmartLLM")
h3 = _load("SmartH3LLM")


class AudioTests(unittest.TestCase):
    def test_vhs_audio_is_mixed_and_resampled(self):
        waveform = torch.stack(
            (torch.linspace(-1, 1, 8000), torch.linspace(1, -1, 8000))
        ).unsqueeze(0)
        prepared, duration = llm._prepare_comfy_audio(
            {"waveform": waveform, "sample_rate": 8000}
        )
        self.assertEqual(tuple(prepared.shape), (1, 16000))
        self.assertAlmostEqual(duration, 1.0)
        self.assertEqual(
            llm._audio_cache_key({"waveform": waveform, "sample_rate": 8000})[:2],
            ((1, 2, 8000), 8000),
        )

    def test_malformed_audio_is_rejected(self):
        with self.assertRaisesRegex(ValueError, r"\[B,C,T\]"):
            llm._prepare_comfy_audio(
                {"waveform": torch.zeros(2, 20), "sample_rate": 16000}
            )

    def test_audio_is_present_in_chat_and_processor_arguments(self):
        class FakeProcessor:
            def __init__(self):
                self.messages = None
                self.kwargs = None
                self.template_kwargs = None

            def apply_chat_template(self, messages, **kwargs):
                self.messages = messages
                self.template_kwargs = kwargs
                return "rendered chat"

            def __call__(self, **kwargs):
                self.kwargs = kwargs
                return {"input_ids": torch.zeros(1, 1, dtype=torch.long)}

        processor = FakeProcessor()
        waveform = torch.zeros(1, 16000)
        llm._prepare_processor_inputs(
            processor,
            "system",
            "analyze",
            [],
            audio_waveform=waveform,
            enable_thinking=False,
        )
        content_types = [item["type"] for item in processor.messages[-1]["content"]]
        self.assertEqual(content_types, ["audio", "text"])
        self.assertIs(processor.kwargs["audio"], waveform)
        self.assertFalse(processor.template_kwargs["enable_thinking"])


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        self.image = torch.zeros(1, 2, 2, 3)

    def test_required_base_images_and_t2va_ignores_images(self):
        self.assertEqual(h3._active_image_tensors("base", "T2VA", [self.image] * 4), [])
        with self.assertRaisesRegex(ValueError, "I2VA requires image_1"):
            h3._active_image_tensors("base", "I2VA", [None] * 4)
        with self.assertRaisesRegex(ValueError, "FL2VA requires image_2"):
            h3._active_image_tensors("base", "FL2VA", [self.image, None, None, None])
        self.assertEqual(
            len(h3._active_image_tensors("base", "L2VA", [self.image, None, None, None])),
            1,
        )

    def test_ref_images_are_numbered_densely(self):
        active = h3._active_image_tensors(
            "ref2VA", "T2VA", [self.image, None, self.image, None]
        )
        self.assertEqual(len(active), 2)
        roles = h3._active_ref_image_roles(
            [self.image, None, self.image, None],
            ["Subject/reference", "First frame", "Last frame", "Auto from prompt"],
        )
        self.assertEqual(roles, ["Subject/reference", "Last frame"])
        inventory = h3._asset_inventory(
            "ref2VA",
            "T2VA",
            2,
            True,
            True,
            3.5,
            ref_image_roles=roles,
        )
        self.assertIn("Picture 1", inventory)
        self.assertIn("Picture 2", inventory)
        self.assertIn("subject/reference source", inventory)
        self.assertIn("concrete last frame", inventory)
        self.assertIn("Video 1", inventory)
        self.assertIn("Audio 1", inventory)

    def test_only_matching_base_example_is_injected(self):
        prompt = h3._generation_system_prompt("base", "I2VA")
        self.assertIn("### Case 2: I2VA", prompt)
        self.assertNotIn("### Case 1: T2VA", prompt)
        self.assertNotIn("### Case 3: FL2VA", prompt)


class ValidationTests(unittest.TestCase):
    def test_all_base_workflow_shapes_validate(self):
        body = (
            "integrated_multimodal_description: [Shot 1] Cinematic, a static shot "
            "shows the requested action.\n\n"
            "overall_soundscape: Quiet room tone.\n\n"
            "non_diegetic_music: N/A"
        )
        prompts = {
            "T2VA": body,
            "I2VA": (
                "For the target video, at 0.00 seconds into the target video, "
                "<Picture 1> (from [Shot 1]) is fully referenced.\n\n" + body
            ),
            "FL2VA": (
                "How the reference pictures align with the target video — Picture 1 "
                "(from Shot 1) aligns with the 0.00-second mark of the target video; "
                "Picture 2 (from Shot 1) aligns with the 15.00-second mark of the target video.\n\n"
                + body.replace("requested action", "path from Picture 1 to Picture 2")
            ),
            "L2VA": (
                "How the reference pictures align with the target video — <Picture 1> "
                "(from [Shot 1]) aligns with the 15.00-second mark of the target video.\n\n"
                + body.replace("requested action", "action converging onto <Picture 1>")
            ),
        }
        for workflow, text in prompts.items():
            with self.subTest(workflow=workflow):
                self.assertEqual(
                    h3._validate_h3_output(text, "base", workflow, 15.0, "", 0), []
                )

    def test_timestamp_and_dialogue_failures_are_reported(self):
        text = (
            "integrated_multimodal_description: [Shot 1] A static shot. "
            "[Shot 2] At 00:15.000, the camera cuts to another view.\n\n"
            "overall_soundscape: Quiet room tone.\n\nnon_diegetic_music: N/A"
        )
        errors = h3._validate_h3_output(
            text, "base", "T2VA", 15.0, "Keep this exact!", 2
        )
        self.assertTrue(any("before the target duration" in error for error in errors))
        self.assertTrue(any("Verbatim" in error for error in errors))

    def test_shot_parser_ignores_in_body_shot_references(self):
        body = (
            "detailed_description:\n"
            "Cinematic style.\n"
            "[Shot 1] The composition follows <Picture 1> and references the framing planned "
            "for [Shot 2] without creating a cut.\n\n"
            "overall_soundscape:\nQuiet room tone.\n\n"
            "non_diegetic_music:\nN/A"
        )
        headers, cuts = h3._shot_headers(h3._description_body(body, "ref2VA"))
        self.assertEqual(headers, [1])
        self.assertEqual(cuts, [])

    def test_concrete_picture_role_definition_is_restored(self):
        text = (
            "subject_definitions:\n"
            "<Subject 1> comes from <Picture 1>.\n\n"
            "summary:\n[keyframe completion] A subject appears.\n\n"
            "retention_analysis:\n...\n\n"
            "detailed_description:\n[Shot 1] The subject appears.\n\n"
            "overall_soundscape:\nN/A\n\n"
            "non_diegetic_music:\nN/A"
        )
        repaired = h3._ensure_picture_role_definitions(text, ["First frame"], 1)
        self.assertIn(
            "<Picture 1> is the first frame of [Shot 1]",
            repaired,
        )

    def test_cleanup_removes_wrapper_but_keeps_spec(self):
        cleaned = h3._sanitize_output(
            "Here is the prompt:\n```text\nintegrated_multimodal_description: [Shot 1] X\n"
            "\noverall_soundscape: N/A\n\nnon_diegetic_music: N/A\n```",
            "base",
        )
        self.assertTrue(cleaned.startswith("integrated_multimodal_description:"))
        self.assertNotIn("```", cleaned)

    def test_ref_cleanup_canonicalizes_reference_labels(self):
        cleaned = h3._sanitize_output(
            "subject_definitions:\n"
            "Picture 1 is a first frame.\n"
            "<picture2> is a last frame from subject 3.",
            "ref2VA",
        )
        self.assertIn("<Picture 1>", cleaned)
        self.assertIn("<Picture 2>", cleaned)
        self.assertIn("<Subject 3>", cleaned)

    def test_ref2va_requires_every_connected_asset_label(self):
        text = (
            "subject_definitions:\n"
            "<Subject 1> is the object from <Picture 1>.\n\n"
            "summary:\n[reference generation] <Subject 1> appears.\n\n"
            "retention_analysis:\n"
            "<Subject 1> (appears in [Shot 1]): fully_preserved - retained.\n\n"
            "detailed_description:\n"
            "Cinematic style.\n[Shot 1] <Subject 1> remains visible.\n\n"
            "overall_soundscape:\nQuiet room tone.\n\n"
            "non_diegetic_music:\nN/A"
        )
        errors = h3._validate_h3_output(
            text,
            "ref2VA",
            "T2VA",
            15.0,
            "",
            0,
            expected_picture_count=1,
            expect_audio=True,
            required_ref_tasks=("keyframe completion", "audio reference"),
            expected_picture_roles=["Last frame"],
        )
        self.assertTrue(any("<Audio 1>" in error for error in errors))
        self.assertFalse(any("Connected reference <Picture 1> is missing" in error for error in errors))
        self.assertTrue(any("keyframe completion" in error for error in errors))
        self.assertTrue(any("`Last frame` role" in error for error in errors))


class PipelineTests(unittest.TestCase):
    def test_analysis_is_passed_to_generation_and_returned_separately(self):
        valid = (
            "integrated_multimodal_description: [Shot 1] Cinematic, a static shot "
            "shows a red ball rolling.\n\n"
            "overall_soundscape: The ball rolls softly.\n\n"
            "non_diegetic_music: N/A"
        )
        calls = []

        def fake_generate(*args, **kwargs):
            calls.append(kwargs)
            return "PICTURES\nA factual red ball." if len(calls) == 1 else valid

        with (
            mock.patch.object(h3, "_validate_model_folder", return_value="model"),
            mock.patch.object(h3, "_load_model", return_value=(object(), object())),
            mock.patch.object(h3, "_generate_text", side_effect=fake_generate),
        ):
            prompt, analysis = h3.SmartH3LLM().run(
                model_folder="model",
                skill="base",
                base_workflow="T2VA",
                ref_image_1_role="Auto from prompt",
                ref_image_2_role="Auto from prompt",
                ref_image_3_role="Auto from prompt",
                ref_image_4_role="Auto from prompt",
                ref_video_role="Auto from prompt",
                prompt="A red ball rolls.",
                verbatim_dialogue="",
                video_duration=15.0,
                visual_style="Cinematic",
                shot_count=0,
                audio_usage="Auto from prompt",
                max_tokens=4096,
                attn_implementation="sdpa",
                device_placement="cuda",
                enable_thinking=False,
                unload_model=False,
                video_fps=30.0,
                max_video_frames=32,
            )
        self.assertEqual(prompt, valid)
        self.assertEqual(analysis, "PICTURES\nA factual red ball.")
        self.assertIn("A factual red ball.", calls[1]["user_prompt"])
        self.assertIsNone(calls[1].get("audio_waveform"))
        self.assertFalse(calls[0]["do_sample"])
        self.assertFalse(calls[1]["do_sample"])
        self.assertFalse(calls[0]["enable_thinking"])
        self.assertFalse(calls[1]["enable_thinking"])

    def test_invalid_draft_can_use_second_text_only_repair(self):
        invalid = (
            "integrated_multimodal_description: [Shot 1] X\n\n"
            "non_diegetic_music: N/A"
        )
        valid = (
            "integrated_multimodal_description: [Shot 1] A static shot.\n\n"
            "overall_soundscape: Quiet room tone.\n\n"
            "non_diegetic_music: N/A"
        )
        results = iter(("UNCERTAINTIES\nNone.", invalid, invalid, valid))
        calls = []

        def fake_generate(*args, **kwargs):
            calls.append(kwargs)
            return next(results)

        with (
            mock.patch.object(h3, "_validate_model_folder", return_value="model"),
            mock.patch.object(h3, "_load_model", return_value=(object(), object())),
            mock.patch.object(h3, "_generate_text", side_effect=fake_generate),
        ):
            prompt, _ = h3.SmartH3LLM().run(
                model_folder="model",
                skill="base",
                base_workflow="T2VA",
                ref_image_1_role="Auto from prompt",
                ref_image_2_role="Auto from prompt",
                ref_image_3_role="Auto from prompt",
                ref_image_4_role="Auto from prompt",
                ref_video_role="Auto from prompt",
                prompt="A static shot.",
                verbatim_dialogue="",
                video_duration=15.0,
                visual_style="Auto",
                shot_count=0,
                audio_usage="Ignore",
                max_tokens=4096,
                attn_implementation="sdpa",
                device_placement="cuda",
                enable_thinking=False,
                unload_model=False,
                video_fps=30.0,
                max_video_frames=32,
            )
        self.assertEqual(prompt, valid)
        self.assertEqual(len(calls), 4)
        self.assertIn("Missing required field `overall_soundscape:`", calls[2]["user_prompt"])
        self.assertIn("USER'S CREATIVE INTENT", calls[2]["user_prompt"])
        self.assertNotIn("pil_images", calls[2])
        self.assertNotIn("pil_video", calls[2])
        self.assertNotIn("audio_waveform", calls[2])
        self.assertNotIn("pil_images", calls[3])

    def test_explicit_ref_roles_become_required_summary_tasks(self):
        tasks = h3._required_ref_task_types(
            image_roles=["First frame", "Subject/reference", "Last frame"],
            has_video=True,
            has_audio=True,
            ref_video_role="Video editing",
            audio_usage="Reference only",
        )
        self.assertEqual(
            tasks,
            (
                "keyframe completion",
                "reference generation",
                "video editing",
                "audio reference",
            ),
        )
        generation_prompt = h3._generation_user_prompt(
            "ref2VA",
            "T2VA",
            "15.00",
            "assets",
            "analysis",
            "intent",
            "controls",
            tasks,
        )
        self.assertIn(
            "keyframe completion + reference generation + video editing + audio reference",
            generation_prompt,
        )


if __name__ == "__main__":
    unittest.main()
