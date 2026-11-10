# PathEval: Learned Visual Grounding and Structured Reasoning

CS 5787 Deep Learning final project. **Can learned visual grounding and structured reasoning improve VLM path evaluation, and which component contributes more?** We build on Aghzal et al., *Evaluating Vision-Language Models as Evaluators in Path Planning* (CVPR 2025).

## Team and ownership

| Member | NetID | Owned contribution |
| --- | --- | --- |
| Gaurav Patel | gp438 | Qwen2-VL inference and three prompting strategies; dataset adapter and combined pipeline integration |
| Yu-Jye Hu | yh2444 | CLIP feature extraction, linear/MLP grounding method, and training experiments |
| Zoe Krishtul | zrk7 | Image-only visual scaffolding, ID-aligned accuracy/invalid-answer evaluation, and integrated prototype run |

Yu-Jye's original exploration is preserved in `notebooks/Clip+ MLP.ipynb`. Gaurav adapted it into `src/grounding.py`; differences are documented there. Zoe's `evaluate()` supplies core metrics; paired McNemar comparisons and swap reporting are implemented in the combined pipeline.

## Working milestone prototype

Three conditions completed on the same 15 2D test examples in Kaggle, each also evaluated with paths swapped. This demonstrates execution, not a completed benchmark or a reliable accuracy improvement.

| Condition | Original accuracy | Swapped accuracy | Same underlying path after swap | Invalid (original) |
| --- | ---: | ---: | ---: | ---: |
| Direct | 53.3% (8/15) | 60.0% (9/15) | 13.3% (2/15) | 0% |
| Structured + CLIP | 53.3% (8/15) | 66.7% (10/15) | 46.7% (7/15) | 0% |
| Direct + scaffolding | 60.0% (9/15) | 66.7% (10/15) | 53.3% (8/15) | 0% |

This table is transcribed from the team's Kaggle output; raw predictions/configurations are not yet included in Git. Both original-layout comparisons against direct have exact McNemar p = 1.000. See [results, provenance, and limitations](docs/milestone-results.md). Gaurav's separate 150-example direct baseline is documented in [prompting details](docs/prompting.md); it is a different run, not a paired comparison with this table.

## Task, method, and metrics

For example i, the inputs are candidate path images x_i1 and x_i2 and scenario s_i; the reference y_i is 1 or 2. The task is to select the preferred path under the scenario.

- **Prompting:** frozen Qwen2-VL-7B-Instruct receives the scenario and images, using direct, generic step-by-step, or task-structured instructions and greedy generation.
- **Grounding:** frozen CLIP ViT-B/32 encodes each half. Normalized features are concatenated as z_i = [CLIP(x_i1); CLIP(x_i2)]. A linear head or two-layer MLP predicts relative path length and sharp-turn count (including sharp-turn ties). Heads train with cross-entropy on training data, with grouped validation, early stopping, and head selection. Predicted attributes become text evidence for Qwen.
- **Scaffolding:** split the combined image into separate ordered path views, using no reference labels or ground-truth descriptors.
- **Evaluation:** accuracy = correct / all evaluated examples; invalid answers count as incorrect and are reported separately. Swapped runs flip labels. Swap consistency measures choosing the same underlying path in both layouts. Exact paired McNemar tests compare correctness with direct prompting.

Ground-truth descriptors are reserved for grounding supervision and a separately named oracle condition. The current heads cover two attributes, not all descriptors. The integrated milestone run covers 2D data; 3D evaluation remains future work. See [pipeline conditions](docs/pipeline.md).

## Run on Kaggle

1. Import [notebooks/kaggle_pipeline.ipynb](notebooks/kaggle_pipeline.ipynb). Enable Internet and GPU T4 x2.
2. For a small integrated check, replace its settings cell with:

   ```python
   REPO = "zoekrishtul/Deep-Learning-Final-Project"
   BRANCH = "main"
   CONDITIONS = ["direct", "structured+clip", "direct+scaffold"]
   RUN_SWAPPED = True
   N_PER_SCENARIO = 1
   N_TRAIN = 500
   BATCH_SIZE = 1
   OUT_DIR = "/kaggle/working/milestone_check_01"
   ```

3. Run cells in order: clone, install dependencies, load/download test data, train grounding, load Qwen, run conditions, and report.
4. Download the entire folder named by `OUT_DIR` before stopping. Keep predictions, `.config.json` files, grounding metrics/checkpoint, attribute predictions, and `comparison.csv`.

No prepared Kaggle dataset is required: the notebook downloads PathEval when attached data is absent. Public GitHub cloning works without a GitHub token; optional credentials belong in Kaggle Secrets. First-time data/model downloads can take time.

The notebook's study defaults run all eight conditions on 150 examples; use the smaller settings above for a check. **Use a new output directory when changing code, data, models, or settings.** Resume currently skips existing IDs without validating configuration compatibility.

## Local setup and execution

Use a virtual environment (Python 3.10/3.11 suggested). Install a hardware-appropriate PyTorch build, then the remaining requirements:

```sh
python -m venv .venv
# macOS/Linux:
source .venv/bin/activate
# Windows PowerShell instead:
# .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

GPU-free checks use synthetic examples and fake models:

```sh
python -m unittest discover tests -v
```

Grounding training tests require PyTorch; tests also need Pillow and scikit-learn. These checks do not establish real-model quality or GPU compatibility. Dependencies are not version-locked; preserve the working environment with run evidence.

On a CUDA machine with adequate memory, run from the repository root:

```sh
python -c "from src.data import fetch_patheval as f; f('train', 'data/patheval', limit=500); f('test', 'data/patheval')"
python -m src.grounding train --manifest data/patheval/train_2D.jsonl --data-root data/patheval --out models/grounding.pt
python -m src.grounding predict --manifest data/patheval/test_2D.jsonl --data-root data/patheval --model models/grounding.pt --n-per-scenario 1 --out results/milestone_check_01/attributes_test.jsonl
python -m src.pipeline run --manifest data/patheval/test_2D.jsonl --data-root data/patheval --attributes results/milestone_check_01/attributes_test.jsonl --n-per-scenario 1 --batch-size 1 --conditions direct structured+clip direct+scaffold --results-dir results/milestone_check_01
```

`run` also writes the comparison. To regenerate it:

```sh
python -m src.pipeline report --results-dir results/milestone_check_01 --conditions direct structured+clip direct+scaffold
```

## Repository map

| Path | Purpose |
| --- | --- |
| `src/data.py` | Manifests, downloads, subsets, image loading/swapping |
| `src/prompting.py` | Qwen inference, prompts, parsing, saved predictions/configs |
| `src/grounding.py` | CLIP features, head training, attribute predictions |
| `src/visual_scaffolding.py` | Separate ordered path views |
| `src/evaluation.py` | ID checks, accuracy, invalid-answer metrics |
| `src/pipeline.py` | Conditions, swap comparisons, exact McNemar test, reports |
| `tests/` | Synthetic/fake-model checks |
| `notebooks/` | Integrated runner and earlier contribution notebooks |
| `docs/` | Technical guides and preliminary results |
| `examples/` | Existing PathEval inspection images |
| `inspect_dataset.py` | Optional exploratory download/scaffolding script |

Full datasets, generated outputs, weights, caches, and credentials are excluded from Git. Small summaries are retained in `docs/`.

## Remaining work and submission

- Expand matched evaluation and separate reasoning/grounding effects with additional conditions.
- Audit path/image overlap across train, validation, and test; grouping by descriptors alone does not prove image independence.
- Require identical unique IDs before trusting reports: current comparisons intersect IDs and can hide incomplete or duplicate records.
- Grounding confidence is an uncalibrated softmax score. Oracle evidence also uses different wording, so its gap does not isolate attribute-error effects alone.
- Retain raw predictions, sample IDs, exact model/package revisions, and configurations from Kaggle.
- Submit the separate 2–3-page ACM milestone report to Gradescope and confirm the assigned TA/grader has repository access. This README is not the required report.

## References and attribution

- Aghzal, M., Yue, X., Plaku, E., and Yao, Z. (2025). *Evaluating Vision-Language Models as Evaluators in Path Planning*. CVPR. [Paper](https://doi.org/10.1109/CVPR52734.2025.00646)
- [Original PathEval code](https://github.com/MohamedAghzal/PathEval) and [dataset](https://huggingface.co/datasets/maghzal/PathEval)
- [Qwen2-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct) and [CLIP ViT-B/32](https://huggingface.co/openai/clip-vit-base-patch32)

Upstream prompt wording and descriptor definitions are attributed in `src/prompting.py` and `docs/prompting.md`. Example images originate from PathEval. Upstream code, data, and model terms remain applicable; this cleanup assigns no new project license.
