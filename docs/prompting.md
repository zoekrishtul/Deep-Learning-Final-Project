# VLM prompting and inference (Gaurav)

Runs frozen **Qwen2-VL-7B-Instruct** on PathEval path pairs with different prompting strategies,
saves one prediction per sample and condition, and gives a quick sanity summary.
Official metrics and paired comparisons are in `src/evaluation.py` (Zoe).

## Files

| File | What it does |
| --- | --- |
| `src/prompting.py` | Prompts, answer parsing, the Qwen2-VL wrapper (batched greedy generation), the run loop (resume + run config), quick summary, CLI |
| `src/data.py` | Converts the released PathEval JSON to the team manifest, fixes image paths, stratified subsets, image loading and left/right swapping |
| `notebooks/kaggle_data_setup.ipynb` | Downloads the 2D test images and writes `test_2D.jsonl` (no GPU) |
| `notebooks/kaggle_prompting.ipynb` | Clones this repo on Kaggle and runs the prompting conditions (GPU T4 x2) |
| `tests/test_prompting.py` | Tests that need no GPU: parsing, prompts, path fixes, swapping, resume, summary math |

## Strategies

| Strategy | Status | Idea |
| --- | --- | --- |
| `direct` | done | Scenario + layout, ask for `Answer: Path 1 or Path 2` (prompt from the original PathEval code) |
| `step_by_step` | done | Generic chain-of-thought: describe each path, reason, then answer. No task-specific structure, so it isolates the effect of reasoning before answering |
| `structured` | done | Task-specific reasoning mirroring the paper's decomposition: (1) which descriptors matter for the scenario and whether lower/higher is better, (2) compare the paths on each from the image, (3) decide. Gives descriptor *definitions*, never their values |

The exact prompt text is in `src/prompting.py` (`BODIES and build_prompt`) and is saved with every run in the config file.

Every strategy can also be combined with Zoe's visual scaffolding (`--scaffold-dir`: the two paths shown as
separate images) and with grounding evidence (`--grounding clip --attributes FILE`, or `--grounding oracle`).
See `docs/pipeline.md` for the full set of conditions.

Every strategy can also be run with `--swap`: the left and right halves of the image are swapped and the label is
flipped. A model that judges path content picks the same underlying path in both layouts; a position-biased model
picks the same side.

## How to run

**On Kaggle (needs a GPU).** Open `notebooks/kaggle_prompting.ipynb`, set Accelerator to *GPU T4 x2*, add the
`patheval-2d-test` dataset, and add a Kaggle secret `GITHUB_TOKEN` (classic token with `repo` scope). Pick the
strategies in the settings cell (default: all three, each also swapped) and run all.

**From the command line (any CUDA machine with ≥16 GB of GPU memory in total):**

```sh
python -m src.prompting run --manifest data/test_2D.jsonl --data-root data/ \
    --strategy direct --n-per-scenario 10 --out results/direct_2D.jsonl
python -m src.prompting run --manifest data/test_2D.jsonl --data-root data/ \
    --strategy direct --n-per-scenario 10 --swap --out results/direct_2D_swapped.jsonl
python -m src.prompting summarize results/direct_2D.jsonl \
    --swapped results/direct_2D_swapped.jsonl --per-scenario
```

`--n-per-scenario 10` picks the same 10 examples from each of the 15 scenarios (150 total, seed 0); omit it to run
all 1050. Rerunning the same command resumes where it stopped.

**Tests (no GPU, e.g. on a Mac):** `python -m unittest discover tests -v`

## Output

`results/<condition>.jsonl`: one row per sample, following the team contract:

```json
{"sample_id": "2383", "condition": "direct", "prediction": 1, "raw_response": "Answer: Path 1.\nExplanation: ...",
 "label": 1, "swapped": false, "scenario": "..."}
```

`prediction` is 1, 2, or `null` when no answer could be parsed (counted as wrong, never guessed). `prediction` and
`label` are relative to the image *as shown*, so swapped rows carry the flipped label. Run settings (model revision,
dtype, versions, GPUs, git commit, prompt template, generation settings, the exact sample IDs) are saved to
`results/<condition>.jsonl.config.json`.

## Preliminary results (2D test, 10 per scenario = 150 samples)

From the Kaggle prototype notebook (same model, prompt, subset, and decoding as `src/prompting.py`):

| Condition | Accuracy | Predicted Path 1 / Path 2 | Invalid |
| --- | --- | --- | --- |
| direct | 50.7% | 27 / 123 | 0% |
| direct, swapped | 54.0% | 26 / 124 | 0% |

Labels are balanced (73 / 77), so always answering "Path 2" scores 51.3%. On the swap test the model picks the same
**side** 78% of the time and the same **path** only 22% of the time: direct prompting is driven by position, not path
content. This matches the paper's finding that low-level perception is the bottleneck, and it gives later conditions
(reasoning prompts, learned grounding) a second target besides accuracy: consistency under swapping.

## Design decisions

- **Greedy decoding** (`do_sample=False`) so results are repeatable.
- **float16** on Kaggle T4s, which lack fast bfloat16; the model is split across both GPUs (~8 GB each).
- **Batching** (8 per batch) halves the time per sample (4.0 s → 2.0 s) with identical answers on a check of 8 samples.
  Left padding is required so every prompt ends where generation starts.
- **Parsing** takes the *last* line with `Answer: Path N`, since reasoning prompts think aloud before answering. Lines
  that name both paths (e.g. the model copying the format line `Answer: Path 1 or Answer: Path 2`) are skipped.
  If there is no usable `Answer:` line, it falls back to the path only if exactly one path is mentioned; otherwise
  the answer is invalid (`null`).
- **Token budgets per strategy** (`DEFAULT_MAX_NEW_TOKENS`): 256 for direct, 768 for step-by-step, 1024 for
  structured. A reasoning answer cut off before its final line would parse as invalid, so a rising invalid rate
  is the signal to raise the budget.
- **All strategies use the same final-answer format** so they are parsed identically and compared fairly.
- **No descriptors in prompts** for visual-only conditions. Descriptors stay in the manifest for evaluation and for a
  separately named oracle condition.
- **Data quirks** (see `src/data.py`): one test record's image path is missing its subfolder; `combined_images_2`
  duplicates `combined_images`; the HuggingFace field names differ from the original GitHub code, so the original
  `qwen_zeroshot.py` cannot run on this release unchanged.

## Sources

- Direct prompt wording and the zero-shot setup: [PathEval `zero-shot/qwen_zeroshot.py`](https://github.com/MohamedAghzal/PathEval/blob/main/zero-shot/qwen_zeroshot.py)
  (Aghzal et al., CVPR 2025). Changes from that script: loader for the HuggingFace release, batched generation,
  robust parsing with invalid-answer tracking, resumable runs with saved configs, swap test.
- Model: [Qwen/Qwen2-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct), loaded with Hugging Face `transformers`.
