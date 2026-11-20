# Final runs: one notebook per person

The full experiment (8 conditions × original/swapped images × 1,050 2D test examples) is too long for one
Kaggle session (12-hour limit) or one account (about 30 GPU hours a week). It is split into three
notebooks that run at the same time on three accounts, with no overlap:

| Notebook | Person | Jobs | Estimated GPU hours |
| --- | --- | --- | --- |
| `run_gaurav.ipynb` | Gaurav | `direct`, `step_by_step`, `structured` (original + swapped) | 13–23 |
| `run_yujye.ipynb` | Yu-Jye | grounding; `direct+clip`, `structured+clip` (original + swapped); `structured+oracle` (original) | 13–25 |
| `run_zoe.ipynb` | Zoe | `direct+scaffold`, `structured+scaffold` (original + swapped); `structured+oracle` (swapped) | 12–24 |

Only `direct`'s speed is measured (about 2 s per answer). Reasoning conditions are estimates; each run's
progress line shows the real seconds per answer and time left.

## How to run

1. Kaggle: *File → Import Notebook* → your notebook. Accelerator **GPU T4 x2**, Internet **On**.
2. *Add-ons → Secrets*: `GITHUB_TOKEN` (classic GitHub token with `repo` scope), ticked for this notebook.
   Optional: `HF_TOKEN` for faster downloads.
3. **Save Version → Save & Run All.** Background runs keep going when you close the tab; an open
   interactive session would use up GPU hours while idle.
4. If Kaggle stops the run (12-hour limit), **do Save & Run All again**. It continues where it stopped.
   Repeat until the "Status" cell prints that everything is complete.

## Where results go

Results are pushed automatically to the **`final-results`** branch, folder **`final_results/2D_full/`**:
every 10 minutes while running and after each finished job. Nothing needs downloading. Each notebook only
writes its own files (`src/results_sync.py`), so the three can push at the same time safely. At most about
10 minutes of work is lost if a session is cut off.

The last cell of each notebook prints a progress table over everyone's saved results.

## Final comparison table

When all three are complete, from the repository root:

```sh
git fetch origin final-results
git worktree add ../final-results origin/final-results
python -m src.pipeline report --results-dir ../final-results/final_results/2D_full
```

This prints the comparison and writes `comparison.csv` / `comparison.json` in that folder.

## Rules

- Don't change `JOBS` in a notebook: that is what prevents overlap.
- Don't change settings (sample size, batch size, model) mid-experiment: runs resume from saved files and
  do not check that settings match. To change settings, use a new `RESULTS_SUBDIR`.
