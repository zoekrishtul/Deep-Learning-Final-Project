# PathEval team project

Research question: Can learned visual grounding and structured reasoning jointly improve VLM path evaluation, and which component contributes more?

This is a lightweight starter. Model loading, training, image transformations, dataset adaptation, and evaluation are not implemented yet. The existing root README is preserved.

## Responsibilities

| Owner | Module | Scope |
| --- | --- | --- |
| Gaurav | src/prompting.py | Frozen Qwen2-VL inference; direct, generic step-by-step, and structured prompting |
| Yu-Jye | src/grounding.py | Frozen CLIP features; linear head versus two-layer MLP; predicted relative path attributes |
| Zoe | src/visual_scaffolding.py | Reproducible visual scaffolding baseline |
| Zoe | src/evaluation.py | Selection accuracy, paired comparisons, subgroup analysis |

## Setup

Use Python 3.10 or 3.11 in a virtual environment. From the repository root:

```sh
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# macOS/Linux:
# source .venv/bin/activate
python -m pip install -r requirements.txt
python -c "from src import prompting, grounding, visual_scaffolding, evaluation; print('Starter imports OK')"
```

The requirements are an initial dependency list, not a tested lockfile. Select the PyTorch build appropriate for your CPU/CUDA environment using the official PyTorch installation instructions before installing the remaining requirements. Qwen2-VL-7B inference needs suitable compute; creating this starter does not download weights. Record exact dependency versions, model revision, device, seed, and generation settings with each experiment.

## Layout

```text
docs/README.md             Project guide
requirements.txt          Shared starting dependencies
src/__init__.py
src/prompting.py           Gaurav (guide: docs/prompting.md)
src/data.py                PathEval manifest adapter, downloads, and image loading (Gaurav)
src/pipeline.py            Runs all conditions and compares them (guide: docs/pipeline.md)
tests/                     GPU-free tests for prompting, data, grounding, and the pipeline
src/grounding.py           Yu-Jye
src/visual_scaffolding.py  Zoe
src/evaluation.py          Zoe
data/README.md             Local dataset instructions
results/README.md          Prediction/output conventions
notebooks/README.md        Exploratory notebooks
notebooks/kaggle_*.ipynb   Kaggle: data setup, prompting runner, full pipeline
notebooks/Clip+ MLP.ipynb  Yu-Jye's grounding exploration (module: src/grounding.py)
```

## Shared integration contract (proposed)

Adapt the released dataset into a local JSONL manifest with one row per example:
`sample_id` (stable string), `split` (train/validation/test), `image_paths` (ordered list of local paths), `scenario` (string), `label` (integer 1 or 2). Optional metadata: `dimension` (2D/3D), criterion, and difficulty when actually available.

Keep path ordering identical across labels, images, prompts, and predictions. This is our proposed internal format, not a claim about the upstream schema. Confirm the released fields and supervision before writing the adapter. Ground-truth descriptors belong in training/evaluation metadata, never in visual-only model inputs.

Predictions use one JSONL row per sample and condition:
`sample_id`, `condition`, `prediction` (1, 2, or null for an invalid/unparsed answer), and `raw_response`. Store run settings separately. Preserve invalid responses and report their rate; agree on the accuracy denominator before comparisons.

## First milestones

1. Verify dataset fields, labels, licensing, and attribute supervision; agree on fixed splits and stable IDs.
2. Run a small direct-prompt baseline and save predictions using the shared contract.
3. Implement evaluation independently using saved predictions.
4. Train linear/MLP grounding heads on training data only; freeze CLIP and Qwen.
5. Compare prompting-only, learned grounding, combined, and visual-scaffolding conditions on matched held-out examples.

Avoid overlap between training and held-out path pairs/images. Fit preprocessing on training data only. Keep ground-truth attribute inputs, if used, in a separately named oracle condition. Scaffolding must use images alone and must not reveal the correct answer. Plan paired bootstrap confidence intervals or McNemar comparisons and 2D/3D breakdowns after the basic evaluation works.

## Team workflow

Create a short feature branch, work in your assigned module, and open a pull request. Agree on interface changes together. Do not commit datasets, model weights, credentials, or generated experiment outputs. Commit small summaries/configurations intentionally when useful.

## References

- [Original PathEval repository](https://github.com/MohamedAghzal/PathEval)
- [PathEval dataset](https://huggingface.co/datasets/maghzal/PathEval)
- Aghzal, Yue, Plaku, and Yao (CVPR 2025), *Evaluating Vision-Language Models as Evaluators in Path Planning*. [Paper](https://doi.org/10.1109/CVPR52734.2025.00646)

No upstream code or data is vendored here. Consult upstream licenses before reuse.
