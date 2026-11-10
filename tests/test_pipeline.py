"""Tests for combining the modules: grounding, scaffolding, prompting, and evaluation.

No GPU or model downloads: Qwen and CLIP are replaced by small fakes.
Run from the repository root:  python -m unittest discover tests -v
"""

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from src import data, grounding, pipeline, prompting
from src.evaluation import evaluate

HAS_TORCH = importlib.util.find_spec("torch") is not None


def make_row(i, label, length=(10, 20), turns=(0, 1), scenario="Scenario."):
    p1 = {"Path length": length[0], "Sharp turns": turns[0]}
    p2 = {"Path length": length[1], "Sharp turns": turns[1]}
    return data.to_manifest_row(i, f"img/{i}_2D.png", scenario, label, p1, p2)


class RecordingEvaluator:
    """Fake Qwen: answers 'Path 1' when told Path 1 is shorter, else 'Path 2'.

    Records what it was shown so tests can check images and prompts.
    """

    def __init__(self):
        self.calls = []

    def generate(self, prompts, images, max_new_tokens=256):
        self.calls.append((prompts, images))
        return ["Answer: Path 1" if "Path 1 is shorter" in p else "Answer: Path 2" for p in prompts]

    def describe(self):
        return {"model": "fake"}


# --------------------------------------------------------------------------
# Grounding (no torch needed)
# --------------------------------------------------------------------------

class TestGroundingLabels(unittest.TestCase):
    def test_attribute_label(self):
        self.assertEqual(grounding.attribute_label(make_row(1, 1, length=(10, 20)), "Path length"), 0)
        self.assertEqual(grounding.attribute_label(make_row(1, 1, length=(30, 20)), "Path length"), 1)
        self.assertIsNone(grounding.attribute_label(make_row(1, 1, length=(5, 5)), "Path length"))
        self.assertEqual(grounding.attribute_label(make_row(1, 1, turns=(2, 2)), "Sharp turns"), 2)

    def test_oracle_attributes_and_swap(self):
        row = make_row(1, 1, length=(10, 20), turns=(1, 1))
        a = grounding.oracle_attributes(row)
        self.assertEqual(a["Path length"]["lower"], "Path 1")
        self.assertEqual(a["Sharp turns"]["lower"], "Tie")
        s = grounding.oracle_attributes(row, swap=True)
        self.assertEqual(s["Path length"]["lower"], "Path 2")     # Path 1 is now on the right
        self.assertEqual(s["Sharp turns"]["lower"], "Tie")

    def test_group_split_keeps_duplicates_together(self):
        # 40 distinct pairs + 10 exact duplicates of the first 10.
        rows = [make_row(i, 1, length=(i, 100 - i), turns=(i % 3, 1)) for i in range(40)]
        rows += [make_row(100 + i, 2, length=(i, 100 - i), turns=(i % 3, 1)) for i in range(10)]
        train, val = grounding.group_split(rows)
        key = lambda r: (r["descriptors"]["path1"]["Path length"], r["descriptors"]["path1"]["Sharp turns"])
        self.assertFalse({key(r) for r in train} & {key(r) for r in val})
        self.assertEqual(len(train) + len(val), 50)
        train2, _ = grounding.group_split(rows)
        self.assertEqual([r["sample_id"] for r in train], [r["sample_id"] for r in train2])


@unittest.skipUnless(HAS_TORCH, "needs torch")
class TestGroundingTraining(unittest.TestCase):
    """Train both head kinds on synthetic features where the attribute is learnable."""

    def test_heads_learn_and_predict_with_swap(self):
        import torch

        class FakeEncoder:
            model_name = "fake-clip"

            def encode_rows(self, rows, data_root, **_):
                # Feature 0 of each half encodes that path's length: learnable signal.
                out = torch.zeros(len(rows), 2, 8)
                for n, r in enumerate(rows):
                    out[n, 0, 0] = r["descriptors"]["path1"]["Path length"] / 100
                    out[n, 1, 0] = r["descriptors"]["path2"]["Path length"] / 100
                    out[n, 0, 1] = r["descriptors"]["path1"]["Sharp turns"]
                    out[n, 1, 1] = r["descriptors"]["path2"]["Sharp turns"]
                return out

        g = torch.Generator().manual_seed(0)
        rows = []
        for i in range(200):
            a, b = torch.randint(10, 90, (2,), generator=g).tolist()
            if a == b:
                b += 1
            ta, tb = torch.randint(0, 3, (2,), generator=g).tolist()
            rows.append(make_row(i, 1, length=(a, b), turns=(ta, tb)))

        bundle = grounding.train_grounding(rows, None, FakeEncoder())
        self.assertEqual(set(bundle["heads"]), set(grounding.ATTRIBUTES))
        self.assertEqual(len(bundle["comparison"]), 4)                    # 2 attributes x 2 kinds
        for h in bundle["heads"].values():
            self.assertIn(h["kind"], ("linear", "mlp"))

        test = rows[:20]
        recs = grounding.predict_attributes(test, None, bundle, FakeEncoder())
        self.assertEqual(len(recs), 40)                                    # original + swapped
        by_key = {(r["sample_id"], r["swapped"]): r["attributes"] for r in recs}
        flips = sum(by_key[(r["sample_id"], False)]["Path length"]["lower"] ==
                    {"Path 1": "Path 2", "Path 2": "Path 1"}[by_key[(r["sample_id"], True)]["Path length"]["lower"]]
                    for r in test)
        self.assertGreaterEqual(flips, 15)   # swapping the image mostly flips the prediction

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "g.pt"
            grounding.save_grounding(bundle, path)
            loaded = grounding.load_grounding(path)
            again = grounding.predict_attributes(test, None, loaded, FakeEncoder())
            self.assertEqual(recs, again)

    def test_pair_features_swap(self):
        import torch
        halves = torch.arange(8.0).view(2, 2, 2)                  # [N=2, halves=2, D=2]
        self.assertEqual(grounding.pair_features(halves)[0].tolist(), [0, 1, 2, 3])
        self.assertEqual(grounding.pair_features(halves, swap=True)[0].tolist(), [2, 3, 0, 1])


# --------------------------------------------------------------------------
# Prompts with scaffolding / grounding
# --------------------------------------------------------------------------

class TestPromptVariants(unittest.TestCase):
    def setUp(self):
        self.row = make_row(1, 1)
        self.clip = {"Path length": {"lower": "Path 1", "confidence": 0.78, "source": "clip"},
                     "Sharp turns": {"lower": "Tie", "confidence": 0.6, "source": "clip"}}

    def test_evidence_block(self):
        p = prompting.build_prompt(self.row, "structured", self.clip)
        self.assertIn("Path 1 is shorter (confidence 0.78).", p)
        self.assertIn("same number of sharp turns", p)
        self.assertIn("It can be wrong", p)
        o = prompting.build_prompt(self.row, "direct", grounding.oracle_attributes(self.row))
        self.assertIn("measured exactly", o)
        self.assertNotIn("confidence", o)

    def test_scaffold_layout(self):
        p = prompting.build_prompt(self.row, "direct", scaffold=True)
        self.assertIn(prompting.SCAFFOLD_LAYOUT, p)
        self.assertNotIn("left side", p)

    def test_prompt_never_contains_label_or_raw_values(self):
        for strategy in prompting.STRATEGIES:
            p = prompting.build_prompt(self.row, strategy, grounding.oracle_attributes(self.row))
            self.assertNotIn("20", p)          # raw descriptor values never appear

    def test_condition_names(self):
        self.assertEqual(prompting.condition_name("direct"), "direct")
        self.assertEqual(prompting.condition_name("structured", True, False, "clip"),
                         "structured+clip_swapped")
        self.assertEqual(prompting.condition_name("direct", False, True), "direct+scaffold")
        for name, spec in pipeline.CONDITIONS.items():
            self.assertEqual(prompting.condition_name(spec["strategy"], False, spec.get("scaffold", False),
                                                      spec.get("grounding")), name)


# --------------------------------------------------------------------------
# End to end: all conditions through Zoe's evaluate()
# --------------------------------------------------------------------------

class TestEndToEnd(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        rows = []
        for i in range(8):
            img = Image.new("RGB", (8, 4), (255, 0, 0))
            img.paste((0, 0, 255), (4, 0, 8, 4))                    # left red, right blue
            (self.tmp / "img").mkdir(exist_ok=True)
            img.save(self.tmp / "img" / f"{i}_2D.png")
            shorter_first = i < 4
            rows.append(make_row(i, 1 if shorter_first else 2, scenario=f"S{i % 2}.",
                                 length=(10, 20) if shorter_first else (20, 10)))
        self.manifest = self.tmp / "m.jsonl"
        data.write_jsonl(rows, self.manifest)
        # "CLIP" attribute file: right for the first 6 samples, wrong for the last 2.
        attrs = []
        for r in rows:
            truth = grounding.oracle_attributes(r)
            for swap in (False, True):
                a = {k: dict(v, source="clip", confidence=0.7) for k, v in
                     grounding.oracle_attributes(r, swap).items()}
                if int(r["sample_id"]) >= 6:
                    a["Path length"]["lower"] = {"Path 1": "Path 2", "Path 2": "Path 1"}[a["Path length"]["lower"]]
                attrs.append({"sample_id": r["sample_id"], "swapped": swap, "attributes": a})
        self.attrs = self.tmp / "attrs.jsonl"
        data.write_jsonl(attrs, self.attrs)

    def test_scaffolded_run_shows_two_images_in_order(self):
        ev = RecordingEvaluator()
        out = self.tmp / "r" / "s.jsonl"
        prompting.run(self.manifest, self.tmp, out, ev, scaffold_dir=self.tmp / "scaf")
        _, images = ev.calls[0]
        self.assertEqual(len(images[0]), 2)
        self.assertEqual(images[0][0].getpixel((0, 0)), (255, 0, 0))   # Path 1 (left half) first
        self.assertTrue((self.tmp / "scaf" / "0_2D_path1.png").exists())  # written by Zoe's scaffold()

        ev2 = RecordingEvaluator()
        prompting.run(self.manifest, self.tmp, self.tmp / "r" / "s2.jsonl", ev2,
                      swap=True, scaffold_dir=self.tmp / "scaf")
        self.assertEqual(ev2.calls[0][1][0][0].getpixel((0, 0)), (0, 0, 255))  # swapped: Path 2 first

    def test_clip_grounding_requires_complete_attributes(self):
        partial = self.tmp / "partial.jsonl"
        data.write_jsonl(data.read_jsonl(self.attrs)[:3], partial)
        with self.assertRaises(ValueError):
            prompting.run(self.manifest, self.tmp, self.tmp / "x.jsonl", RecordingEvaluator(),
                          grounding="clip", attributes_path=partial)

    def test_all_conditions_and_report(self):
        results = self.tmp / "results"
        outputs = pipeline.run_conditions(list(pipeline.CONDITIONS), self.manifest, self.tmp, results,
                                          RecordingEvaluator(), attributes_path=self.attrs)
        self.assertEqual(len(outputs), 2 * len(pipeline.CONDITIONS))
        rows = {r["condition"]: r for r in pipeline.compare(results, list(pipeline.CONDITIONS))}

        # The fake always answers "Path 2" without evidence: 4/8 right, never consistent.
        self.assertAlmostEqual(rows["direct"]["accuracy"], 0.5)
        self.assertAlmostEqual(rows["direct"]["consistent"], 0.0)
        # With oracle evidence it follows the truth: perfect and swap-consistent.
        self.assertAlmostEqual(rows["structured+oracle"]["accuracy"], 1.0)
        self.assertAlmostEqual(rows["structured+oracle"]["consistent"], 1.0)
        # With CLIP evidence it is right exactly where the "CLIP" file is right (6/8).
        self.assertAlmostEqual(rows["structured+clip"]["accuracy"], 0.75)
        self.assertIn("vs_baseline_p", rows["structured+clip"])

        table = pipeline.format_table(list(rows.values()))
        self.assertIn("structured+oracle", table)
        pipeline.save_table(list(rows.values()), results / "comparison.csv")
        self.assertTrue((results / "comparison.csv").exists())


class TestEvaluationBridge(unittest.TestCase):
    def test_conversion_matches_zoes_evaluate(self):
        preds = [{"sample_id": "1", "prediction": 1, "label": 1},
                 {"sample_id": "2", "prediction": 2, "label": 1},
                 {"sample_id": "3", "prediction": None, "label": 2}]
        m = evaluate(*pipeline.to_evaluation_format(preds))
        self.assertEqual((m["total"], m["correct"], m["invalid_count"]), (3, 1, 1))

    def test_mcnemar(self):
        self.assertEqual(pipeline.mcnemar_exact([True] * 5, [True] * 5), (0, 0, 1.0))
        b, c, p = pipeline.mcnemar_exact([False] * 5, [True] * 5)      # 5 wins, 0 losses
        self.assertEqual((b, c), (0, 5))
        self.assertAlmostEqual(p, 2 / 32)


if __name__ == "__main__":
    unittest.main()
