"""Gaurav: Qwen2-VL inference and prompting strategies for PathEval.

Usage (from the repository root):

    # Run a condition and save predictions
    python -m src.prompting run --manifest data/test_2D.jsonl --data-root data/ \\
        --strategy direct --n-per-scenario 10 --out results/direct_2D.jsonl

    # Other strategies: --strategy step_by_step | structured
    # Same examples with the left/right halves swapped (position-bias test)
    python -m src.prompting run ... --swap --out results/direct_2D_swapped.jsonl

    # With visual scaffolding (two separate images) and/or grounding evidence
    python -m src.prompting run ... --scaffold-dir results/scaffolded
    python -m src.prompting run ... --grounding clip --attributes results/attributes_test.jsonl

    # Quick sanity summary (official metrics live in src/evaluation.py)
    python -m src.prompting summarize results/direct_2D.jsonl \\
        --swapped results/direct_2D_swapped.jsonl

Design notes:
  * Qwen2-VL is frozen; decoding is greedy (``do_sample=False``) so runs are repeatable.
  * Visual-only conditions never read labels or ground-truth descriptors.
  * Predictions follow the team contract: one JSONL row per sample and condition.
    ``prediction`` and ``label`` are always relative to the image *as shown*, so a
    swapped condition carries the flipped label.
  * Run settings are written next to the predictions as ``<out>.config.json``.
  * Runs resume: samples already in the output file are skipped.
  * torch / transformers are imported lazily, so prompts and parsing can be
    tested on any machine without a GPU.

Based on the zero-shot setup of Aghzal et al. (CVPR 2025),
https://github.com/MohamedAghzal/PathEval/blob/main/zero-shot/qwen_zeroshot.py
(prompt wording for direct prompting and the descriptor definitions used by
the structured prompt are taken from that repository).
"""

import argparse
import json
import os
import platform
import re
import subprocess
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from src.data import load_manifest, load_pair_image, read_jsonl, stratified_subset

STRATEGIES = ("direct", "step_by_step", "structured")
DEFAULT_MODEL = "Qwen/Qwen2-VL-7B-Instruct"

# Reasoning prompts write much more before the answer; too small a budget cuts the
# answer off (it then parses as invalid), so each strategy gets its own default.
DEFAULT_MAX_NEW_TOKENS = {"direct": 256, "step_by_step": 768, "structured": 1024}


# --------------------------------------------------------------------------
# Prompts
# --------------------------------------------------------------------------

LAYOUT = "Path 1 is on the left side and Path 2 is on the right side."
# Visual scaffolding (Zoe, src/visual_scaffolding.py) shows the two paths as separate images.
SCAFFOLD_LAYOUT = "The first image shows Path 1 and the second image shows Path 2."

# Final-answer line shared by the reasoning prompts, so all strategies are parsed the same way.
FINAL_ANSWER = ("After your reasoning, end with one final line in exactly this format:\n"
                "Answer: Path 1   or   Answer: Path 2")

# Every prompt is:  "{scenario} {layout}\n" + [grounding evidence] + strategy body.
# Without evidence and scaffolding, the text is exactly the original single-image prompt.

# 1. Direct: answer immediately (wording from the original PathEval zero-shot code).
DIRECT_BODY = (
    "Your answer should follow the format below:\n"
    "Answer: Path 1 or Path 2.\n"
    "Explanation: Why you chose the path (1 or 2)."
)

# 2. Step-by-step: generic chain-of-thought. Asks the model to reason first but gives
#    no task-specific structure, so it isolates the effect of "thinking before answering".
STEP_BY_STEP_BODY = (
    "Let's think step by step. First describe what you see in each path, "
    "then reason about which path better satisfies the scenario.\n"
    + FINAL_ANSWER
)

# 3. Structured: task-specific reasoning that mirrors the paper's decomposition of
#    PathEval into (a) abstraction: which descriptors the scenario cares about,
#    (b) perception: how the two paths compare on each, (c) integration: the decision.
#    Descriptor definitions are from the original PathEval perception prompt.
DESCRIPTOR_DEFINITIONS = (
    "- Minimum clearance: the smallest distance between the path and the obstacles.\n"
    "- Maximum clearance: the largest distance between the path and the obstacles.\n"
    "- Average clearance: the average distance between the path and the obstacles.\n"
    "- Path length: the total length of the path.\n"
    "- Smoothness: the sum of absolute angles between path segments "
    "(a lower value means a smoother path).\n"
    "- Sharp turns: the number of turns sharper than 90 degrees.\n"
    "- Maximum angle: the sharpest turn angle in the path."
)

STRUCTURED_BODY = (
    "Paths can be described by these properties:\n" + DESCRIPTOR_DEFINITIONS + "\n"
    "Work through the following steps.\n"
    "Step 1 - Relevant properties: list the properties that matter for this scenario, "
    "and for each say whether a lower or a higher value is better.\n"
    "Step 2 - Comparison: for each relevant property, look at the image and state which "
    "path is better (Path 1, Path 2, or about equal), with a short reason.\n"
    "Step 3 - Decision: weigh the comparisons from Step 2 and decide which path better "
    "satisfies the scenario.\n"
    + FINAL_ANSWER
)

BODIES = {
    "direct": DIRECT_BODY,
    "step_by_step": STEP_BY_STEP_BODY,
    "structured": STRUCTURED_BODY,
}

# --- Grounding evidence (attributes from src/grounding.py) -------------------------------

ATTRIBUTE_PHRASES = {
    "Path length": {"Path 1": "Path 1 is shorter", "Path 2": "Path 2 is shorter",
                    "Tie": "Both paths have about the same length"},
    "Sharp turns": {"Path 1": "Path 1 has fewer sharp turns",
                    "Path 2": "Path 2 has fewer sharp turns",
                    "Tie": "Both paths have the same number of sharp turns"},
}
EVIDENCE_HEADERS = {
    "clip": ("A perception model estimated the following from the image. It can be wrong; "
             "the confidence is its estimated probability of being right.\n"),
    "oracle": "The following facts were measured exactly from the paths:\n",
}


def describe_attributes(attributes):
    """Turn grounding attributes into the evidence block of a prompt ('' if none)."""
    if not attributes:
        return ""
    sources = {a.get("source", "clip") for a in attributes.values()}
    if len(sources) != 1 or not sources <= set(EVIDENCE_HEADERS):
        raise ValueError(f"Mixed or unknown attribute sources: {sources}")
    source = sources.pop()
    lines = []
    for attr, a in attributes.items():
        phrases = ATTRIBUTE_PHRASES.get(attr)
        text = (phrases[a["lower"]] if phrases else
                f"{a['lower']} has the lower {attr.lower()}" if a["lower"] != "Tie" else
                f"Both paths have about the same {attr.lower()}")
        lines.append(f"- {text}" + (f" (confidence {a['confidence']:.2f})." if source == "clip" else "."))
    return EVIDENCE_HEADERS[source] + "\n".join(lines) + "\n"


def build_prompt(sample, strategy="direct", attributes=None, scaffold=False):
    """Build the text prompt for one sample.

    Uses only the scenario text, plus grounding ``attributes`` when given (predicted, or
    ground truth for the oracle condition). Labels are never used.
    """
    if strategy not in STRATEGIES:
        raise ValueError(f"Unknown prompting strategy: {strategy!r}")
    layout = SCAFFOLD_LAYOUT if scaffold else LAYOUT
    return f"{sample['scenario']} {layout}\n" + describe_attributes(attributes) + BODIES[strategy]


# --------------------------------------------------------------------------
# Answer parsing
# --------------------------------------------------------------------------

_ANSWER_RE = re.compile(r"Answer\s*[:：]\s*[*_\s]*(?:Path\s*)?([12])\b", re.IGNORECASE)
_PATH_RE = re.compile(r"Path\s*([12])\b", re.IGNORECASE)


def parse_answer(text):
    """Extract the chosen path (1 or 2) from model output; None if unclear.

    1. Use the LAST line containing "Answer: Path N" (reasoning prompts think aloud
       before the final answer). Lines naming both paths, e.g. a copied format line
       "Answer: Path 1 or Answer: Path 2", are skipped as ambiguous.
    2. Otherwise, if the text mentions only one of Path 1 / Path 2, use it.
    3. Otherwise return None (counted as invalid, never guessed).
    """
    if not text:
        return None
    for line in reversed(text.splitlines()):
        answers = set(_ANSWER_RE.findall(line))
        if not answers:
            continue
        if len(answers | set(_PATH_RE.findall(line))) == 1:
            return int(answers.pop())
    mentioned = set(_PATH_RE.findall(text))
    if len(mentioned) == 1:
        return int(mentioned.pop())
    return None


def flip(path):
    """1 <-> 2; None stays None. Maps answers between original and swapped images."""
    return {1: 2, 2: 1}.get(path)


# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------

class QwenPathEvaluator:
    """Frozen Qwen2-VL with batched, greedy generation."""

    def __init__(self, model_name=DEFAULT_MODEL, dtype="float16", device_map="auto"):
        import torch
        from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

        self.torch = torch
        self.model_name = model_name
        self.dtype = dtype
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_name, torch_dtype=getattr(torch, dtype), device_map=device_map)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False
        self.processor = AutoProcessor.from_pretrained(model_name)
        # Pad on the left so every prompt ends right where generation starts.
        self.processor.tokenizer.padding_side = "left"

    def generate(self, prompts, images, max_new_tokens=256):
        """Generate one response per prompt.

        ``images[i]`` is one PIL image, or a list of images shown in that order
        (e.g. Path 1 then Path 2 for visual scaffolding).
        """
        texts, flat_images = [], []
        for prompt, imgs in zip(prompts, images):
            imgs = imgs if isinstance(imgs, (list, tuple)) else [imgs]
            flat_images.extend(imgs)
            messages = [{"role": "user", "content": [{"type": "image"} for _ in imgs] +
                         [{"type": "text", "text": prompt}]}]
            texts.append(self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True))
        inputs = self.processor(text=texts, images=flat_images, padding=True,
                                return_tensors="pt").to(self.model.device)
        with self.torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=max_new_tokens,
                                      do_sample=False, temperature=None, top_p=None, top_k=None)
        new_tokens = out[:, inputs["input_ids"].shape[1]:]
        return self.processor.batch_decode(new_tokens, skip_special_tokens=True)

    def describe(self):
        """Settings worth recording with every run."""
        import torch
        import transformers

        gpus = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
        return {
            "model": self.model_name,
            "model_revision": getattr(self.model.config, "_commit_hash", None),
            "dtype": self.dtype,
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "gpus": gpus,
        }


# --------------------------------------------------------------------------
# Prediction
# --------------------------------------------------------------------------

GROUNDING_SOURCES = ("clip", "oracle")


def condition_name(strategy, swap=False, scaffold=False, grounding=None):
    """e.g. 'direct', 'structured+clip', 'direct+scaffold', 'structured+oracle_swapped'."""
    name = strategy + ("+scaffold" if scaffold else "") + (f"+{grounding}" if grounding else "")
    return f"{name}_swapped" if swap else name


def make_record(sample, condition, swap, raw_response):
    """One prediction row in the team contract format (plus convenience fields)."""
    return {
        "sample_id": sample["sample_id"],
        "condition": condition,
        "prediction": parse_answer(raw_response),
        "raw_response": raw_response,
        # Convenience fields for quick checks; evaluation should join on the manifest.
        "label": flip(sample["label"]) if swap else sample["label"],
        "swapped": swap,
        "scenario": sample["scenario"],
    }


def load_scaffolded_images(paths, swap=False):
    """Load Zoe's separate (Path 1, Path 2) images; swapped means Path 2 is shown first."""
    from PIL import Image

    imgs = []
    for p in paths:
        with Image.open(p) as img:
            imgs.append(img.convert("RGB"))
    return imgs[::-1] if swap else imgs


def scaffold_samples(samples, data_root, out_dir):
    """Split images with Zoe's ``scaffold()`` (once) -> {sample_id: (path1_file, path2_file)}."""
    from src.visual_scaffolding import scaffold

    sources = {s["sample_id"]: Path(data_root) / s["image_paths"][0] for s in samples}
    expected = {sid: (Path(out_dir) / f"{src.stem}_path1.png", Path(out_dir) / f"{src.stem}_path2.png")
                for sid, src in sources.items()}
    missing = [sid for sid, (a, b) in expected.items() if not (a.exists() and b.exists())]
    if missing:
        written = scaffold([str(sources[sid]) for sid in missing], out_dir)
        for sid, w in zip(missing, written):
            assert Path(w["path1"]) == expected[sid][0], "unexpected scaffold file name"
    return expected


def predict(samples, evaluator, data_root, strategy="direct", swap=False, max_new_tokens=None,
            scaffold_paths=None, attributes=None, grounding=None):
    """Predict a batch of samples; returns prediction records in input order.

    scaffold_paths: {sample_id: (path1_file, path2_file)} to show two separate images.
    attributes:     {(sample_id, swapped): attribute dict} grounding evidence for the prompt.
    """
    max_new_tokens = max_new_tokens or DEFAULT_MAX_NEW_TOKENS[strategy]
    scaffold = scaffold_paths is not None
    condition = condition_name(strategy, swap, scaffold, grounding)
    prompts, images = [], []
    for s in samples:
        attrs = attributes[(s["sample_id"], swap)] if attributes is not None else None
        prompts.append(build_prompt(s, strategy, attrs, scaffold))
        images.append(load_scaffolded_images(scaffold_paths[s["sample_id"]], swap) if scaffold
                      else load_pair_image(s, data_root, swap))
    responses = evaluator.generate(prompts, images, max_new_tokens)
    return [make_record(s, condition, swap, r) for s, r in zip(samples, responses)]


def grounding_attributes(samples, grounding, attributes_path=None, swap=False):
    """Collect the evidence for each sample: learned ('clip') or ground truth ('oracle')."""
    if grounding == "oracle":
        from src.grounding import oracle_attributes
        return {(s["sample_id"], swap): oracle_attributes(s, swap) for s in samples}
    if grounding == "clip":
        if not attributes_path:
            raise ValueError("grounding='clip' needs attributes_path (from src.grounding predict)")
        from src.grounding import load_attribute_file
        attrs = load_attribute_file(attributes_path)
        missing = [s["sample_id"] for s in samples if (s["sample_id"], swap) not in attrs]
        if missing:
            raise ValueError(f"{len(missing)} samples have no predicted attributes "
                             f"(swapped={swap}), e.g. {missing[:3]}")
        return attrs
    raise ValueError(f"Unknown grounding source: {grounding!r}")


def _git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


def run(manifest, data_root, out, evaluator, strategy="direct", swap=False,
        n_per_scenario=None, seed=0, batch_size=8, max_new_tokens=None,
        scaffold_dir=None, grounding=None, attributes_path=None):
    """Run one condition over a manifest (or a stratified subset) and save predictions.

    scaffold_dir: show Path 1 and Path 2 as separate images, split by Zoe's scaffold()
                  into this folder.
    grounding:    add attribute evidence to the prompt: "clip" (predictions from
                  ``attributes_path``) or "oracle" (ground truth, upper bound).
    Resumes if ``out`` already exists. Writes run settings to ``<out>.config.json``.
    """
    if strategy not in STRATEGIES:
        raise ValueError(f"Unknown prompting strategy: {strategy!r}")
    if grounding not in (None,) + GROUNDING_SOURCES:
        raise ValueError(f"Unknown grounding source: {grounding!r}")
    max_new_tokens = max_new_tokens or DEFAULT_MAX_NEW_TOKENS[strategy]
    rows = load_manifest(manifest)
    samples = stratified_subset(rows, n_per_scenario, seed) if n_per_scenario else rows

    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = {r["sample_id"] for r in read_jsonl(out)} if out.exists() else set()
    todo = [s for s in samples if s["sample_id"] not in done]
    scaffold_paths = scaffold_samples(todo, data_root, scaffold_dir) if scaffold_dir else None
    attributes = grounding_attributes(samples, grounding, attributes_path, swap) if grounding else None

    config = {
        "condition": condition_name(strategy, swap, bool(scaffold_dir), grounding),
        "strategy": strategy,
        "swap": swap,
        "scaffold": bool(scaffold_dir),
        "grounding": grounding,
        "attributes_path": str(attributes_path) if attributes_path else None,
        "prompt_template": build_prompt({"scenario": "{scenario}"}, strategy,
                                        scaffold=bool(scaffold_dir)),
        "manifest": str(manifest),
        "n_samples": len(samples),
        "n_per_scenario": n_per_scenario,
        "seed": seed,
        "sample_ids": [s["sample_id"] for s in samples],
        "batch_size": batch_size,
        "generation": {"do_sample": False, "max_new_tokens": max_new_tokens},
        "git_commit": _git_commit(),
        "python": platform.python_version(),
        "started_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        **evaluator.describe(),
    }
    with open(f"{out}.config.json", "w") as f:
        json.dump(config, f, indent=2)

    print(f"{config['condition']}: {len(samples)} samples, {len(done)} already done, "
          f"{len(todo)} to run")
    start = time.time()
    with open(out, "a") as f:
        for i in range(0, len(todo), batch_size):
            batch = todo[i:i + batch_size]
            for record in predict(batch, evaluator, data_root, strategy, swap, max_new_tokens,
                                  scaffold_paths, attributes, grounding):
                f.write(json.dumps(record) + "\n")
            f.flush()
            n = i + len(batch)
            rate = (time.time() - start) / n
            print(f"  {n}/{len(todo)}  {rate:.1f}s/sample  "
                  f"~{rate * (len(todo) - n) / 60:.0f} min left", end="\r")
    print(f"\nDone in {time.time() - start:.0f}s -> {out}")
    return out


# --------------------------------------------------------------------------
# Quick summary (sanity check only; official metrics are in src/evaluation.py)
# --------------------------------------------------------------------------

def summarize(path, swapped_path=None, per_scenario=False):
    """Return a dict of quick metrics for a prediction file."""
    rows = read_jsonl(path)
    n = len(rows)
    summary = {
        "n": n,
        "accuracy": sum(r["prediction"] == r["label"] for r in rows) / n,  # invalid = wrong
        "invalid_rate": sum(r["prediction"] is None for r in rows) / n,
        "predictions": dict(Counter(r["prediction"] for r in rows)),
        "labels": dict(Counter(r["label"] for r in rows)),
    }
    if per_scenario:
        hits = defaultdict(list)
        for r in rows:
            hits[r["scenario"]].append(r["prediction"] == r["label"])
        summary["per_scenario"] = {s: sum(h) / len(h) for s, h in hits.items()}
    if swapped_path:
        orig = {r["sample_id"]: r for r in rows}
        swap = {r["sample_id"]: r for r in read_jsonl(swapped_path)}
        ids = [i for i in orig if i in swap]
        summary["swap"] = {
            "n_paired": len(ids),
            "accuracy_swapped": sum(swap[i]["prediction"] == swap[i]["label"] for i in ids) / len(ids),
            # Same underlying path chosen in both layouts -> judging content.
            "consistent": sum(orig[i]["prediction"] is not None and
                              orig[i]["prediction"] == flip(swap[i]["prediction"])
                              for i in ids) / len(ids),
            # Same side chosen in both layouts -> judging position.
            "same_side": sum(orig[i]["prediction"] is not None and
                             orig[i]["prediction"] == swap[i]["prediction"]
                             for i in ids) / len(ids),
        }
    return summary


def print_summary(s):
    print(f"n = {s['n']}")
    print(f"accuracy      {s['accuracy']:.1%}   (invalid answers count as wrong)")
    print(f"invalid rate  {s['invalid_rate']:.1%}")
    print(f"predictions   {s['predictions']}")
    print(f"labels        {s['labels']}")
    if "per_scenario" in s:
        print("per scenario:")
        for sc, acc in sorted(s["per_scenario"].items(), key=lambda kv: -kv[1]):
            print(f"  {acc:6.1%}  {sc[:70]}...")
    if "swap" in s:
        w = s["swap"]
        print(f"swap test ({w['n_paired']} paired):")
        print(f"  accuracy swapped   {w['accuracy_swapped']:.1%}")
        print(f"  consistent (same path, content-driven)  {w['consistent']:.1%}")
        print(f"  same side  (position-driven)            {w['same_side']:.1%}")


# --------------------------------------------------------------------------
# Command line
# --------------------------------------------------------------------------

def main(argv=None):
    parser = argparse.ArgumentParser(description="Qwen2-VL prompting on PathEval")
    sub = parser.add_subparsers(dest="command", required=True)

    r = sub.add_parser("run", help="run a prompting condition")
    r.add_argument("--manifest", required=True)
    r.add_argument("--data-root", required=True, help="folder that image_paths are relative to")
    r.add_argument("--out", required=True, help="output predictions .jsonl")
    r.add_argument("--strategy", default="direct", choices=STRATEGIES)
    r.add_argument("--swap", action="store_true", help="swap left/right halves (and labels)")
    r.add_argument("--n-per-scenario", type=int, default=None, help="default: all samples")
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--batch-size", type=int, default=8)
    r.add_argument("--max-new-tokens", type=int, default=None,
                   help="default depends on strategy: " + str(DEFAULT_MAX_NEW_TOKENS))
    r.add_argument("--scaffold-dir", default=None,
                   help="show the two paths as separate images (Zoe's scaffold) cached here")
    r.add_argument("--grounding", default=None, choices=GROUNDING_SOURCES,
                   help="add attribute evidence: clip (needs --attributes) or oracle")
    r.add_argument("--attributes", default=None, help="attribute file from src.grounding predict")
    r.add_argument("--model", default=DEFAULT_MODEL)
    r.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32"])

    s = sub.add_parser("summarize", help="quick summary of a predictions file")
    s.add_argument("predictions")
    s.add_argument("--swapped", default=None, help="swapped-condition predictions to pair with")
    s.add_argument("--per-scenario", action="store_true")

    args = parser.parse_args(argv)
    if args.command == "run":
        evaluator = QwenPathEvaluator(args.model, args.dtype)
        run(args.manifest, args.data_root, args.out, evaluator, args.strategy, args.swap,
            args.n_per_scenario, args.seed, args.batch_size, args.max_new_tokens,
            args.scaffold_dir, args.grounding, args.attributes)
    else:
        print_summary(summarize(args.predictions, args.swapped, args.per_scenario))


if __name__ == "__main__":
    main()
