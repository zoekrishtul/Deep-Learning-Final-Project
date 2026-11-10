
"""Zoe: shared held-out evaluation.

Validate unique IDs, align predictions with references,
and report accuracy and invalid-answer rate.
"""


def evaluate(predictions, references):
    """Evaluate predictions against references using dataset IDs."""

    # Build dictionaries indexed by example ID
    pred_by_id = {}
    ref_by_id = {}

    for item in predictions:
        example_id = str(item["id"])

        if example_id in pred_by_id:
            raise ValueError(f"Duplicate prediction ID: {example_id}")

        pred_by_id[example_id] = item["prediction"]

    for item in references:
        example_id = str(item["id"])

        if example_id in ref_by_id:
            raise ValueError(f"Duplicate reference ID: {example_id}")

        ref_by_id[example_id] = item["ground_truth"]

    # Ensure we're evaluating the same examples
    if pred_by_id.keys() != ref_by_id.keys():
        raise ValueError("Prediction and reference IDs do not match.")

    if not ref_by_id:
        raise ValueError("Cannot evaluate an empty dataset.")

    correct = 0
    invalid = 0

    for example_id, actual in ref_by_id.items():
        prediction = pred_by_id[example_id]

        if prediction not in ("Path 1", "Path 2"):
            invalid += 1
        elif prediction == actual:
            correct += 1

    total = len(ref_by_id)

    return {
        "total": total,
        "correct": correct,
        "accuracy": correct / total,
        "invalid_count": invalid,
        "invalid_rate": invalid / total,
    }
