# Preliminary milestone results

## Integrated 15-example check

Source: the Kaggle comparison table shared by Zoe during the milestone run. This is a manual transcription, not a fresh evaluation during repository cleanup. Raw predictions and run configurations have not been added to Git; retain them from Kaggle to verify exact IDs and settings.

The selected settings were one test example per scenario (15 total), three conditions, and original/swapped layouts. The shared settings specified 500 grounding training records and batch size 1. Saved run configurations remain the authoritative record.

| Condition | n | Original accuracy | Swapped accuracy | Swap consistency | Picks Path 2 (original) | Invalid (original) | Wins/losses vs direct | Exact McNemar p |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| direct | 15 | 53.3% (8/15) | 60.0% (9/15) | 13.3% (2/15) | 93% (14/15) | 0.0% | baseline | -- |
| structured+clip | 15 | 53.3% (8/15) | 66.7% (10/15) | 46.7% (7/15) | 40% (6/15) | 0.0% | +4 / -4 | 1.000 |
| direct+scaffold | 15 | 60.0% (9/15) | 66.7% (10/15) | 53.3% (8/15) | 73% (11/15) | 0.0% | +2 / -1 | 1.000 |

Consistency means choosing the same underlying path after swapping, not the same position. It does not measure correctness itself. Swapped accuracy uses flipped reference labels; wins/losses and p-values compare original-layout correctness.

## Interpretation

- All three conditions completed real-model runs. Original-layout answers were all parseable.
- Direct chose Path 2 on 14/15 original examples, indicating a strong position preference on this subset.
- Scaffolding obtained one additional correct original-layout answer. Structured + CLIP changed eight correctness outcomes with no net accuracy gain.
- Grounding and scaffolding increased observed swap consistency on this small subset.
- Neither reported accuracy comparison is significant under the exact McNemar test. These exploratory results do not establish a reliable accuracy gain.
- Structured + CLIP changes both prompting and evidence relative to direct. Add structured-only and direct+CLIP before attributing changes to either component.

## Separate earlier baseline

[Prompting documentation](prompting.md) separately records Gaurav's 150-example direct baseline: 50.7% original accuracy, 54.0% swapped accuracy, and 22% swap consistency. It is a different run and sample size; do not combine it with the 15-example table as a paired comparison.

## Evidence to retain

Download the entire `/kaggle/working/smoke_test_01/` folder from the completed run: all six prediction JSONL files, their configurations, `comparison.csv`, `grounding_comparison.json`, and `attributes_test.jsonl`. Retain the grounding checkpoint separately. Save/export the executed notebook with outputs.

For the report, label these as preliminary results and discuss larger matched evaluation, reasoning/grounding ablations, split-overlap checks, and raw-output/configuration validation as next steps.
