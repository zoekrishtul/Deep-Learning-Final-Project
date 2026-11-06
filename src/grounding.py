"""Yu-Jye: learned visual grounding with frozen CLIP features.

TODO: verify attribute supervision; compare a linear head with a two-layer MLP.
Train heads on the training split only. Keep CLIP and Qwen frozen.
"""


def predict_attributes(image_paths):
    """Predict relative path attributes from ordered images using a trained head."""
    raise NotImplementedError("TODO (Yu-Jye): implement CLIP + linear/MLP grounding.")
