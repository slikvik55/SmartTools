from __future__ import annotations

import importlib.util
import sys
import types
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = "_smarttools_h3_prompt_tests"
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


_load("SmartLLM")
_load("SmartH3LLM")
h3_prompt = _load("SmartH3Prompt")


def _inputs(**overrides):
    values = {
        "skill": "base",
        "base_workflow": "T2VA",
        "ref_image_1_role": "Auto from prompt",
        "ref_image_2_role": "Auto from prompt",
        "ref_image_3_role": "Auto from prompt",
        "ref_image_4_role": "Auto from prompt",
        "ref_video_role": "Auto from prompt",
        "picture_1": "",
        "picture_2": "",
        "picture_3": "",
        "picture_4": "",
        "video_1": "",
        "audio_1": "",
        "shot_1": "A medium-wide static shot shows a red ball rolling across a wooden floor.",
        "shot_2": "",
        "shot_2_start_time": 5.0,
        "shot_3": "",
        "shot_3_start_time": 10.0,
        "verbatim_dialogue": "",
        "video_duration": 15.0,
        "visual_style": "Cinematic",
        "audio_usage": "Ignore",
        "subject_definitions": "",
        "summary": "",
        "retention_analysis": "",
        "overall_soundscape": "",
        "non_diegetic_music": "",
    }
    values.update(overrides)
    return values


class BaseWorkflowTests(unittest.TestCase):
    def test_all_base_workflows_build_valid_shapes(self):
        cases = {
            "T2VA": {},
            "I2VA": {"picture_1": "A woman beside a rain-covered train window."},
            "FL2VA": {
                "picture_1": "A closed umbrella beside a bicycle.",
                "picture_2": "The same umbrella open above the cyclist.",
            },
            "L2VA": {"picture_1": "A broken glass settled on a dark floor."},
        }
        for workflow, extra in cases.items():
            with self.subTest(workflow=workflow):
                prompt = h3_prompt.SmartH3Prompt().run(
                    **_inputs(base_workflow=workflow, **extra)
                )[0]
                self.assertIn("integrated_multimodal_description:", prompt)
                self.assertIn("overall_soundscape:", prompt)
                self.assertIn("non_diegetic_music:", prompt)
                if workflow == "T2VA":
                    self.assertTrue(prompt.startswith("integrated_multimodal_description:"))
                else:
                    self.assertIn("<Picture 1>", prompt)

    def test_three_shots_get_formatted_start_times(self):
        prompt = h3_prompt.SmartH3Prompt().run(
            **_inputs(
                shot_2="The camera cuts to a close-up of the rolling ball.",
                shot_2_start_time=3.25,
                shot_3="The ball comes to rest against a chair leg.",
                shot_3_start_time=9.125,
            )
        )[0]
        self.assertIn("[Shot 2] At 00:03.250,", prompt)
        self.assertIn("[Shot 3] At 00:09.125,", prompt)

    def test_verbatim_dialogue_is_preserved(self):
        prompt = h3_prompt.SmartH3Prompt().run(
            **_inputs(
                shot_1=(
                    "A woman (S1) says, <d>[English] Keep this exactly!</d> "
                    "while facing the camera."
                ),
                verbatim_dialogue="Keep this exactly!",
            )
        )[0]
        self.assertIn("<d>[English] Keep this exactly!</d>", prompt)


class RefWorkflowTests(unittest.TestCase):
    def test_ref_defaults_build_tasks_definitions_and_retention(self):
        prompt = h3_prompt.SmartH3Prompt().run(
            **_inputs(
                skill="ref2VA",
                ref_image_1_role="First frame",
                ref_image_2_role="Subject/reference",
                ref_video_role="Video editing",
                picture_1="A wide opening frame in a bright kitchen.",
                picture_2="A woman with short dark hair and a green jacket.",
                video_1="A handheld source clip with three quick cuts.",
                audio_1="A rhythmic percussion track with kitchen ambience.",
                audio_usage="Copy/reuse",
            )
        )[0]
        self.assertIn("<Picture 1> is the first frame of [Shot 1]", prompt)
        self.assertIn("<Subject 2>", prompt)
        self.assertIn("<Video 1>", prompt)
        self.assertIn("<Audio 1>", prompt)
        self.assertIn(
            "[keyframe completion + reference generation + video editing + audio reuse]",
            prompt,
        )
        self.assertIn("The target video is an edited version of <Video 1>.", prompt)
        self.assertIn("<Audio 1>: partially_copy", prompt)

    def test_custom_section_details_are_preserved(self):
        prompt = h3_prompt.SmartH3Prompt().run(
            **_inputs(
                skill="ref2VA",
                ref_image_1_role="Subject/reference",
                picture_1="A polished red toy robot with square blue eyes.",
                subject_definitions=(
                    "<Subject 1> is the polished red toy robot from <Picture 1>, "
                    "with square blue eyes."
                ),
                summary="<Subject 1> crosses a workshop table in one continuous shot.",
                retention_analysis=(
                    "<Subject 1> (appears in [Shot 1]): fully_preserved - "
                    "the red shell and square blue eyes are retained."
                ),
                overall_soundscape="Small metal feet tap against the wooden table.",
                non_diegetic_music="Sparse marimba notes at a moderate tempo.",
            )
        )[0]
        self.assertIn("polished red toy robot from <Picture 1>", prompt)
        self.assertIn("crosses a workshop table", prompt)
        self.assertIn("Small metal feet tap", prompt)
        self.assertIn("Sparse marimba notes", prompt)


class InputValidationTests(unittest.TestCase):
    def test_sparse_ref_pictures_are_rejected(self):
        with self.assertRaisesRegex(ValueError, r"<Picture 1> is empty"):
            h3_prompt.SmartH3Prompt().run(
                **_inputs(
                    skill="ref2VA",
                    picture_2="A reference entered without Picture 1.",
                )
            )

    def test_shot_three_requires_shot_two(self):
        with self.assertRaisesRegex(ValueError, "Shot 3 cannot be used"):
            h3_prompt.SmartH3Prompt().run(
                **_inputs(shot_3="An invalid third shot.")
            )

    def test_start_times_must_increase_and_fit_duration(self):
        with self.assertRaisesRegex(ValueError, "Shot 3 start time must be later"):
            h3_prompt.SmartH3Prompt().run(
                **_inputs(
                    shot_2="Second shot.",
                    shot_2_start_time=8.0,
                    shot_3="Third shot.",
                    shot_3_start_time=7.0,
                )
            )
        with self.assertRaisesRegex(ValueError, "before video_duration"):
            h3_prompt.SmartH3Prompt().run(
                **_inputs(shot_2="Second shot.", shot_2_start_time=15.0)
            )

    def test_node_mapping_and_widget_labels(self):
        self.assertIs(
            h3_prompt.NODE_CLASS_MAPPINGS["SmartH3Prompt"],
            h3_prompt.SmartH3Prompt,
        )
        required = h3_prompt.SmartH3Prompt.INPUT_TYPES()["required"]
        self.assertEqual(required["picture_1"][1]["label"], "<Picture 1>")
        self.assertNotIn("shot_count", required)
        self.assertNotIn("model_folder", required)


if __name__ == "__main__":
    unittest.main()
