# Combined pipeline

How the three parts of the project fit together, and how to run the full study.

```
              PathEval train images                       PathEval test images
                       │                                           │
                       ▼                                           │
   src/grounding.py (Yu-Jye): frozen CLIP + linear/MLP heads       │
   "which path is shorter / has fewer sharp turns"  ──predicts──►  │
                                                                   ▼
   src/visual_scaffolding.py (Zoe): split into Path 1 / Path 2 ─► src/prompting.py (Gaurav)
                                                                   Qwen2-VL-7B + direct / step-by-step /
                                                                   structured prompts (+ swap test)
                                                                   │
                                                                   ▼
                                       src/pipeline.py  ──►  src/evaluation.py (Zoe)  ──►  comparison table
```

## Conditions

All conditions run on the same test samples (default: 10 per scenario, 150 total), each on the original
image and with the left and right halves swapped.

| Condition | Strategy | Images | Evidence in the prompt | Question it answers |
| --- | --- | --- | --- | --- |
| `direct` | direct | combined | none | Baseline |
| `step_by_step` | step-by-step | combined | none | Does generic reasoning help? |
| `structured` | structured | combined | none | Does task-specific reasoning help? |
| `direct+scaffold` | direct | separate | none | Does showing the paths separately help? (scaffolding baseline) |
| `structured+scaffold` | structured | separate | none | Scaffolding plus reasoning |
| `direct+clip` | direct | combined | CLIP predictions | **Grounding only** |
| `structured+clip` | structured | combined | CLIP predictions | **Grounding + reasoning (combined)** |
| `structured+oracle` | structured | combined | true attributes | Upper bound: how much perfect grounding could help |

The oracle uses the same two attributes as the CLIP heads, so the gap between `structured+clip` and
`structured+oracle` is caused by CLIP's prediction errors alone.

## How to run

**Kaggle (recommended):** `notebooks/kaggle_pipeline.ipynb`. GPU T4 x2, Internet on, attach the
`patheval-2d-test` dataset, add the `GITHUB_TOKEN` secret (and optionally `HF_TOKEN`). Choose conditions in the
settings cell. Reasoning conditions write far more text than direct, so running everything takes several hours.
Use *Save Version → Save & Run All* so it runs in the background, and start with fewer conditions or
`RUN_SWAPPED = False` when time is short. Every run resumes from its output file.

**Command line (any CUDA machine):**
```sh
python -c "from src.data import fetch_patheval as f; f('train', 'data/patheval', limit=500); f('test', 'data/patheval')"
python -m src.grounding train   --manifest data/patheval/train_2D.jsonl --data-root data/patheval --out models/grounding.pt
python -m src.grounding predict --manifest data/patheval/test_2D.jsonl  --data-root data/patheval \
    --model models/grounding.pt --n-per-scenario 10 --out results/attributes_test.jsonl
python -m src.pipeline run --manifest data/patheval/test_2D.jsonl --data-root data/patheval \
    --attributes results/attributes_test.jsonl --n-per-scenario 10
python -m src.pipeline report --results-dir results
```

## Results table

`src/pipeline.py` scores every condition with Zoe's `evaluate()` (unchanged) and adds:

- **acc swapped**: accuracy on the swapped images.
- **consistent**: share of samples where the same *underlying* path is chosen in both layouts. A model
  that judges path content scores high; a model that answers by position scores low.
- **picks P2**: how often the model answers "Path 2" (position bias; the labels are ~50/50).
- **+/- vs direct**: samples this condition gets right that `direct` gets wrong, and vice versa, with an
  exact McNemar test (*p*) on those paired differences.

Saved as `results/comparison.csv`.

## Interfaces between the parts

**Attribute predictions** (`src/grounding.py` → `src/prompting.py`): one JSONL row per sample and layout.

```json
{"sample_id": "2383", "swapped": false,
 "attributes": {"Path length": {"lower": "Path 1", "confidence": 0.78, "head": "mlp", "source": "clip"},
                "Sharp turns": {"lower": "Tie", "confidence": 0.61, "head": "linear", "source": "clip"}}}
```

`lower` is the path with the lower value, relative to the image as shown. For the swapped layout, the CLIP
heads are run again on the swapped halves, rather than flipping the earlier answer.

In the prompt this becomes, for example:
```
A perception model estimated the following from the image. It can be wrong; the confidence is its
estimated probability of being right.
- Path 1 is shorter (confidence 0.78).
- Both paths have the same number of sharp turns (confidence 0.61).
```

**Scaffolding** (`src/visual_scaffolding.py` → `src/prompting.py`): `scaffold()` writes
`<image>_path1.png` and `<image>_path2.png`; Qwen receives both images in order (Path 2 first when swapped),
and the prompt says "The first image shows Path 1 and the second image shows Path 2."

**Evaluation** (`src/prompting.py` → `src/evaluation.py`): `pipeline.to_evaluation_format` converts prediction
rows (`prediction` 1/2/null) into Zoe's format (`id`, `prediction` "Path 1"/"Path 2", `ground_truth`).

## Grounding module

`src/grounding.py` is Yu-Jye's method from `notebooks/Clip+ MLP.ipynb`, moved into a module: frozen CLIP
ViT-B/32 features of each half, concatenated; linear and MLP heads per attribute; grouped train/validation
split; Adam with early stopping. Differences from the notebook are listed at the top of the file. The main
one: every head uses the same early-stopping training loop. Both head kinds are trained and reported
(`grounding_comparison.json`); the one with higher validation accuracy is used for each attribute.

The notebook's validation results for comparison: path length, linear 60.6% vs MLP 79.8%; sharp turns,
linear 58.6% vs MLP 56.6% (majority class 51.5%).

## Tests

`python -m unittest discover tests -v`. Covers grounding labels, grouped splits, training and save/load on
synthetic features, prompts with evidence and scaffolding, two-image input order, and an end-to-end run of
every condition through Zoe's `evaluate()` with fake models. No GPU or downloads needed (PyTorch is needed for
the two training tests; they are skipped without it).
