"""End-to-end experiment: prompting x visual scaffolding x learned grounding.

Ties the team's modules together without changing them:
  * src/prompting.py          (Gaurav)  Qwen2-VL, prompting strategies, swap test
  * src/visual_scaffolding.py (Zoe)     split each image into separate Path 1 / Path 2 images
  * src/grounding.py          (Yu-Jye)  CLIP + linear/MLP attribute predictions
  * src/evaluation.py         (Zoe)     accuracy and invalid rate, used exactly as written

Conditions (all on the same samples; each can also be run with halves swapped):

    name                 strategy      images      evidence in prompt      role in the study
    direct               direct        combined    -                       baseline
    step_by_step         step_by_step  combined    -                       prompting only
    structured           structured    combined    -                       prompting only
    direct+scaffold      direct        separate    -                       scaffolding baseline
    structured+scaffold  structured    separate    -                       scaffolding + reasoning
    direct+clip          direct        combined    CLIP predictions        grounding only
    structured+clip      structured    combined    CLIP predictions        grounding + reasoning
    structured+oracle    structured    combined    true attributes         upper bound for grounding

Usage (from the repository root):
    python -m src.pipeline run    --manifest ... --data-root ... --conditions direct structured+clip \\
                                  --attributes results/attributes_test.jsonl --n-per-scenario 10
    python -m src.pipeline report --results-dir results --conditions direct structured+clip
"""

import argparse
import csv
import json
from math import comb
from pathlib import Path

from src.data import read_jsonl
from src.evaluation import evaluate
from src.prompting import flip, run

CONDITIONS = {
    "direct":              {"strategy": "direct"},
    "step_by_step":        {"strategy": "step_by_step"},
    "structured":          {"strategy": "structured"},
    "direct+scaffold":     {"strategy": "direct", "scaffold": True},
    "structured+scaffold": {"strategy": "structured", "scaffold": True},
    "direct+clip":         {"strategy": "direct", "grounding": "clip"},
    "structured+clip":     {"strategy": "structured", "grounding": "clip"},
    "structured+oracle":   {"strategy": "structured", "grounding": "oracle"},
}


def output_path(results_dir, condition, swap=False, dim="2D"):
    return Path(results_dir) / f"{condition}_{dim}{'_swapped' if swap else ''}.jsonl"


def run_conditions(conditions, manifest, data_root, results_dir, evaluator, swaps=(False, True),
                   attributes_path=None, scaffold_dir=None, **run_kwargs):
    """Run every requested condition (and its swapped version). Returns {(condition, swap): path}."""
    unknown = [c for c in conditions if c not in CONDITIONS]
    if unknown:
        raise ValueError(f"Unknown conditions {unknown}; choose from {list(CONDITIONS)}")
    scaffold_dir = scaffold_dir or Path(results_dir) / "scaffolded"
    outputs = {}
    for name in conditions:
        spec = CONDITIONS[name]
        for swap in swaps:
            outputs[(name, swap)] = run(
                manifest, data_root, output_path(results_dir, name, swap), evaluator,
                strategy=spec["strategy"], swap=swap,
                scaffold_dir=scaffold_dir if spec.get("scaffold") else None,
                grounding=spec.get("grounding"),
                attributes_path=attributes_path if spec.get("grounding") == "clip" else None,
                **run_kwargs)
    return outputs


# --------------------------------------------------------------------------
# Evaluation (Zoe's evaluate) and comparison
# --------------------------------------------------------------------------

def to_evaluation_format(prediction_rows):
    """Our prediction rows -> the (predictions, references) lists Zoe's evaluate() expects.

    Zoe's format: {"id", "prediction": "Path 1"/"Path 2"/anything else = invalid} and
    {"id", "ground_truth": "Path 1"/"Path 2"}. Labels are relative to the image as shown,
    so swapped runs are scored against the flipped label.
    """
    predictions = [{"id": r["sample_id"],
                    "prediction": f"Path {r['prediction']}" if r["prediction"] in (1, 2) else None}
                   for r in prediction_rows]
    references = [{"id": r["sample_id"], "ground_truth": f"Path {r['label']}"}
                  for r in prediction_rows]
    return predictions, references


def mcnemar_exact(correct_a, correct_b):
    """Two-sided exact McNemar test on paired correctness (lists of bools).

    Returns (b, c, p): b = A right & B wrong, c = A wrong & B right.
    """
    b = sum(a and not bb for a, bb in zip(correct_a, correct_b))
    c = sum(bb and not a for a, bb in zip(correct_a, correct_b))
    n = b + c
    if n == 0:
        return b, c, 1.0
    tail = sum(comb(n, k) for k in range(min(b, c) + 1)) / 2 ** n
    return b, c, min(1.0, 2 * tail)


def compare(results_dir, conditions, baseline="direct", dim="2D"):
    """One row per condition: Zoe's metrics, position bias, swap consistency, test vs baseline."""
    def load(name, swap):
        p = output_path(results_dir, name, swap, dim)
        return {r["sample_id"]: r for r in read_jsonl(p)} if p.exists() else None

    base = load(baseline, False)
    rows = []
    for name in conditions:
        orig = load(name, False)
        if orig is None:
            print(f"(skipping {name}: no results yet)")
            continue
        metrics = evaluate(*to_evaluation_format(orig.values()))          # Zoe's evaluation
        preds = [r["prediction"] for r in orig.values()]
        row = {"condition": name, "n": metrics["total"], "accuracy": metrics["accuracy"],
               "invalid_rate": metrics["invalid_rate"],
               "picks_path_2": sum(p == 2 for p in preds) / len(preds)}

        swapped = load(name, True)
        if swapped:
            ids = [i for i in orig if i in swapped]
            row["accuracy_swapped"] = evaluate(*to_evaluation_format(
                [swapped[i] for i in ids]))["accuracy"]
            row["consistent"] = sum(orig[i]["prediction"] in (1, 2) and
                                    orig[i]["prediction"] == flip(swapped[i]["prediction"])
                                    for i in ids) / len(ids)

        if base and name != baseline:
            ids = [i for i in orig if i in base]
            b, c, p = mcnemar_exact([base[i]["prediction"] == base[i]["label"] for i in ids],
                                    [orig[i]["prediction"] == orig[i]["label"] for i in ids])
            row.update({"vs_baseline_wins": c, "vs_baseline_losses": b, "vs_baseline_p": p})
        rows.append(row)
    return rows


def format_table(rows, baseline="direct"):
    cols = [("condition", "condition", "{}"), ("n", "n", "{}"), ("accuracy", "acc", "{:.1%}"),
            ("accuracy_swapped", "acc swapped", "{:.1%}"), ("consistent", "consistent", "{:.1%}"),
            ("picks_path_2", "picks P2", "{:.0%}"), ("invalid_rate", "invalid", "{:.1%}"),
            ("vs_baseline_wins", f"+/- vs {baseline}", None), ("vs_baseline_p", "p", "{:.3f}")]
    header = [c[1] for c in cols]
    lines = []
    for r in rows:
        cells = []
        for key, _, fmt in cols:
            if key == "vs_baseline_wins":
                cells.append(f"+{r[key]} / -{r['vs_baseline_losses']}" if key in r else "")
            else:
                cells.append(fmt.format(r[key]) if key in r else "")
        lines.append(cells)
    widths = [max(len(h), *(len(l[i]) for l in lines)) if lines else len(h)
              for i, h in enumerate(header)]
    out = ["  ".join(h.ljust(w) for h, w in zip(header, widths)),
           "  ".join("-" * w for w in widths)]
    out += ["  ".join(c.ljust(w) for c, w in zip(l, widths)) for l in lines]
    return "\n".join(out)


def save_table(rows, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    keys = list(dict.fromkeys(k for r in rows for k in r))   # all columns, first-seen order
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


# --------------------------------------------------------------------------
# Command line
# --------------------------------------------------------------------------

def main(argv=None):
    parser = argparse.ArgumentParser(description="Run and compare PathEval conditions")
    sub = parser.add_subparsers(dest="command", required=True)

    r = sub.add_parser("run")
    r.add_argument("--manifest", required=True)
    r.add_argument("--data-root", required=True)
    r.add_argument("--results-dir", default="results")
    r.add_argument("--conditions", nargs="+", default=list(CONDITIONS))
    r.add_argument("--attributes", default=None, help="from: python -m src.grounding predict")
    r.add_argument("--no-swap", action="store_true", help="skip the swapped-image runs")
    r.add_argument("--n-per-scenario", type=int, default=None)
    r.add_argument("--batch-size", type=int, default=8)

    c = sub.add_parser("report")
    c.add_argument("--results-dir", default="results")
    c.add_argument("--conditions", nargs="+", default=list(CONDITIONS))
    c.add_argument("--baseline", default="direct")

    args = parser.parse_args(argv)
    if args.command == "run":
        from src.prompting import QwenPathEvaluator
        run_conditions(args.conditions, args.manifest, args.data_root, args.results_dir,
                       QwenPathEvaluator(), swaps=(False,) if args.no_swap else (False, True),
                       attributes_path=args.attributes, n_per_scenario=args.n_per_scenario,
                       batch_size=args.batch_size)
    rows = compare(args.results_dir, args.conditions, getattr(args, "baseline", "direct"))
    print(format_table(rows, getattr(args, "baseline", "direct")))
    save_table(rows, Path(args.results_dir) / "comparison.csv")
    with open(Path(args.results_dir) / "comparison.json", "w") as f:
        json.dump(rows, f, indent=2)


if __name__ == "__main__":
    main()
