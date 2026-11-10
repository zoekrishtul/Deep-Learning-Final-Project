# Project documentation

The current overview, team ownership/NetIDs, formal task and method, setup, and run commands are in the [root README](../README.md).

- [Integrated pipeline and conditions](pipeline.md)
- [Prompting and earlier 150-example direct baseline](prompting.md)
- [Milestone 15-example integrated results](milestone-results.md)
- [Notebook guide](../notebooks/README.md)

## Data contract

src/data.py adapts released PathEval records into a local JSONL manifest. Each row contains sample_id (string), split, dimension, image_paths (one combined image path relative to the data root), scenario, label (1/2), and per-path descriptors.

Prediction rows contain sample_id, condition, prediction (1/2/null), raw_response, label, swapped, and scenario. Labels are relative to the shown layout. Invalid/unparsed predictions count as incorrect.

Labels and ground-truth descriptors must not enter visual-only prompts. Descriptors support training and the separately named oracle condition. Comparisons must use identical unique IDs and preserve path ordering.

## Team workflow

Use feature branches and pull requests. Keep datasets, checkpoints, credentials, and bulk outputs out of Git. Small, attributed result summaries can be committed under docs/. Preserve raw outputs/configurations separately as submission evidence.

The implementation is a working prototype, not a set of placeholder modules. The root README lists known limitations and next steps.
