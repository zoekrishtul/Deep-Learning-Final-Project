"""Tests for the prompting pipeline that run without a GPU or model download.

Run from the repository root:  python -m unittest discover tests -v
"""

import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from src import data, prompting


class FakeEvaluator:
    """Stands in for Qwen2-VL: always answers 'Path 2' (like the real model, mostly)."""

    def generate(self, prompts, images, max_new_tokens=256):
        self.last_images = images
        return ["Answer: Path 2.\nExplanation: fake."] * len(prompts)

    def describe(self):
        return {"model": "fake"}


def make_sample(i, label=1, scenario="Scenario A."):
    return data.to_manifest_row(i, f"img/{i}.png", scenario, label, {"x": 1}, {"x": 2})


class TestParseAnswer(unittest.TestCase):
    def test_standard_format(self):
        self.assertEqual(prompting.parse_answer("Answer: Path 1.\nExplanation: ..."), 1)
        self.assertEqual(prompting.parse_answer("Answer: Path 2."), 2)

    def test_markdown_and_spacing(self):
        self.assertEqual(prompting.parse_answer("**Answer:** Path 2"), 2)
        self.assertEqual(prompting.parse_answer("Answer: **Path 1**"), 1)
        self.assertEqual(prompting.parse_answer("answer:path2"), 2)
        self.assertEqual(prompting.parse_answer("Answer: 1"), 1)

    def test_last_answer_wins(self):
        text = "The format is Answer: Path 1 or Path 2.\nThinking...\nAnswer: Path 2."
        self.assertEqual(prompting.parse_answer(text), 2)

    def test_fallback_single_mention(self):
        self.assertEqual(prompting.parse_answer("I would choose Path 1 here."), 1)

    def test_ambiguous_or_empty_is_none(self):
        self.assertIsNone(prompting.parse_answer("Path 1 is shorter but Path 2 is smoother."))
        self.assertIsNone(prompting.parse_answer("I cannot tell."))
        self.assertIsNone(prompting.parse_answer(""))
        self.assertIsNone(prompting.parse_answer(None))

    def test_no_false_match_on_other_numbers(self):
        self.assertIsNone(prompting.parse_answer("Answer: Path 12"))


class TestPrompts(unittest.TestCase):
    def test_direct_prompt_uses_only_scenario(self):
        sample = make_sample(1)
        prompt = prompting.build_prompt(sample, "direct")
        self.assertTrue(prompt.startswith("Scenario A."))
        self.assertIn("Path 1 is on the left side", prompt)
        self.assertNotIn("descriptors", prompt)

    def test_unknown_strategy(self):
        with self.assertRaises(ValueError):
            prompting.build_prompt(make_sample(1), "magic")

    def test_unimplemented_strategy(self):
        with self.assertRaises(NotImplementedError):
            prompting.build_prompt(make_sample(1), "structured")


class TestData(unittest.TestCase):
    def test_resolve_image_path(self):
        repo = {"2D/combined_images/2/2383_2D.png", "2D/combined_images/2/2_2D.png"}
        self.assertEqual(data.resolve_image_path("combined_images/2/2383_2D.png", repo),
                         "2D/combined_images/2/2383_2D.png")
        self.assertEqual(data.resolve_image_path("combined_images/2_2D.png", repo),
                         "2D/combined_images/2/2_2D.png")
        with self.assertRaises(FileNotFoundError):
            data.resolve_image_path("combined_images/5_2D.png", repo)

    def test_manifest_from_clean_and_release_agree(self):
        clean = [{"id": 2383, "image": "2D/combined_images/2/2383_2D.png", "scenario": "S.",
                  "label": "Path 1", "path1_descriptors": {"a": 1}, "path2_descriptors": {"a": 2}}]
        release = [{"id": 2383, "image": "combined_images/2/2383_2D.png", "scenario": "S.",
                    "ground truth": "Path 1", "path 1 descriptors": {"a": 1},
                    "path 2 descriptors": {"a": 2}}]
        repo = {"2D/combined_images/2/2383_2D.png"}
        a = data.manifest_from_clean(clean)
        b = data.manifest_from_release(release, repo)
        self.assertEqual(a, b)
        self.assertEqual(a[0]["sample_id"], "2383")
        self.assertEqual(a[0]["label"], 1)

    def test_parse_label(self):
        self.assertEqual(data.parse_label("Path 1"), 1)
        self.assertEqual(data.parse_label("Path 2"), 2)

    def test_stratified_subset_is_deterministic_and_balanced(self):
        rows = [make_sample(i, scenario=f"S{i % 3}") for i in range(30)]
        a = data.stratified_subset(rows, 4, seed=0)
        b = data.stratified_subset(rows, 4, seed=0)
        self.assertEqual([r["sample_id"] for r in a], [r["sample_id"] for r in b])
        self.assertEqual(len(a), 12)
        self.assertEqual({r["scenario"] for r in a}, {"S0", "S1", "S2"})

    def test_swap_halves(self):
        img = Image.new("RGB", (4, 2), (255, 0, 0))
        img.paste((0, 0, 255), (2, 0, 4, 2))           # left red, right blue
        swapped = data.swap_halves(img)
        self.assertEqual(swapped.getpixel((0, 0)), (0, 0, 255))
        self.assertEqual(swapped.getpixel((3, 1)), (255, 0, 0))
        self.assertEqual(data.swap_halves(swapped).tobytes(), img.tobytes())


class TestRunAndSummarize(unittest.TestCase):
    """End-to-end on a tiny fake dataset with a fake model."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        rows = []
        for i in range(6):
            Image.new("RGB", (8, 4), (i, i, i)).save(self.tmp / f"{i}.png")
            rows.append(data.to_manifest_row(i, f"{i}.png", f"S{i % 2}.", 1 if i < 3 else 2,
                                             {}, {}))
        self.manifest = self.tmp / "manifest.jsonl"
        data.write_jsonl(rows, self.manifest)

    def test_run_writes_contract_records_and_resumes(self):
        out = self.tmp / "preds.jsonl"
        prompting.run(self.manifest, self.tmp, out, FakeEvaluator(), batch_size=4)
        recs = data.read_jsonl(out)
        self.assertEqual(len(recs), 6)
        for key in ("sample_id", "condition", "prediction", "raw_response"):
            self.assertIn(key, recs[0])
        self.assertEqual(recs[0]["condition"], "direct")
        self.assertTrue((self.tmp / "preds.jsonl.config.json").exists())

        # Second run resumes and adds nothing.
        prompting.run(self.manifest, self.tmp, out, FakeEvaluator(), batch_size=4)
        self.assertEqual(len(data.read_jsonl(out)), 6)

    def test_swap_flips_labels_and_summary(self):
        orig, swp = self.tmp / "o.jsonl", self.tmp / "s.jsonl"
        prompting.run(self.manifest, self.tmp, orig, FakeEvaluator())
        prompting.run(self.manifest, self.tmp, swp, FakeEvaluator(), swap=True)
        swapped = data.read_jsonl(swp)
        self.assertEqual(swapped[0]["condition"], "direct_swapped")
        self.assertEqual(swapped[0]["label"], 2)          # original label 1, flipped

        s = prompting.summarize(orig, swp)
        self.assertAlmostEqual(s["accuracy"], 0.5)          # always "Path 2", 3/6 labels are 2
        self.assertAlmostEqual(s["swap"]["same_side"], 1.0)  # pure position bias
        self.assertAlmostEqual(s["swap"]["consistent"], 0.0)

    def test_config_records_sample_ids(self):
        out = self.tmp / "p.jsonl"
        prompting.run(self.manifest, self.tmp, out, FakeEvaluator(), n_per_scenario=2)
        with open(f"{out}.config.json") as f:
            cfg = json.load(f)
        self.assertEqual(cfg["n_samples"], 4)
        self.assertEqual(len(cfg["sample_ids"]), 4)
        self.assertIn("{scenario}", cfg["prompt_template"])


if __name__ == "__main__":
    unittest.main()
