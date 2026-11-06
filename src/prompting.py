"""Gaurav: Qwen2-VL inference and prompting strategies.

TODO: implement direct, generic step-by-step, and structured prompts.
Keep Qwen frozen and emit records using the shared prediction contract.
"""

STRATEGIES = ("direct", "step_by_step", "structured")


def predict(sample, strategy="direct"):
    """Return a prediction record for a model-safe sample (no labels/descriptors)."""
    if strategy not in STRATEGIES:
        raise ValueError(f"Unknown prompting strategy: {strategy}")
    raise NotImplementedError("TODO (Gaurav): load Qwen2-VL and implement inference.")
