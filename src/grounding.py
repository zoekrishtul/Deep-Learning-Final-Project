"""Yu-Jye: learned visual grounding with frozen CLIP features.

Predicts relative path attributes ("which path is shorter", "which path has fewer
sharp turns") from the image, so the VLM can be given this evidence in its prompt.

Method (designed and validated by Yu-Jye in ``notebooks/Clip+ MLP.ipynb``; moved into
this module unchanged unless noted):
  * Split each side-by-side image into its Path 1 and Path 2 halves.
  * Encode each half with frozen CLIP ViT-B/32, L2-normalize, and concatenate
    [Path 1, Path 2] into a 1024-d pair feature.
  * Train a small head per attribute on the training split only:
      - "linear": Linear(1024, C)
      - "mlp":    Linear(1024, 128) -> ReLU -> Dropout(0.2) -> Linear(128, C)
    Full-batch Adam (lr 1e-3, weight decay 1e-3), up to 200 epochs, early stopping on
    validation loss (patience 20), seed 42. "Sharp turns" is 3-way (Path 1 / Path 2 /
    Tie) with class weights, because ties are the majority class.
  * Validation split: 20% by group, where examples with identical descriptor values
    share a group, so no path pair appears in both train and validation.

Changes from the notebook (Gaurav, when moving it into src/):
  * One training loop for every head. In the notebook, the path-length *linear* head
    ran a fixed 100 epochs without early stopping; here it uses early stopping like
    the others, so linear-vs-MLP is compared under the same procedure.
  * Images are read from a local manifest (``src/data.py``) instead of streaming.
  * Predictions are made for any manifest (e.g. the test set), for the image as shown
    or with halves swapped, and written in the format the prompting code reads.
  * Each attribute uses the head (linear or MLP) with the higher validation accuracy.
    Both are trained and reported, so the linear-vs-MLP comparison is kept.

Usage (from the repository root):
    python -m src.grounding train   --manifest data/train_2D.jsonl --data-root data/patheval \\
                                    --n-train 500 --out models/grounding.pt
    python -m src.grounding predict --manifest data/test_2D.jsonl --data-root data/patheval \\
                                    --model models/grounding.pt --out results/attributes_test.jsonl
"""

import argparse
import copy
import json
from pathlib import Path

from src.data import load_manifest, load_pair_image, read_jsonl, stratified_subset, write_jsonl

CLIP_MODEL = "openai/clip-vit-base-patch32"
SEED = 42

# Attributes the grounding module predicts. "lower" means: which path has the lower value.
ATTRIBUTES = {
    "Path length": {"classes": ("Path 1", "Path 2"), "class_weights": False},
    "Sharp turns": {"classes": ("Path 1", "Path 2", "Tie"), "class_weights": True},
}
# Descriptor values used to group near-duplicate path pairs (same as the notebook).
GROUP_KEYS = ("Path length", "Sharp turns")


# --------------------------------------------------------------------------
# Labels and splits (no torch needed)
# --------------------------------------------------------------------------

def attribute_label(row, attribute):
    """Class index for one attribute: 0 = Path 1 lower, 1 = Path 2 lower, 2 = tie.

    For attributes without a tie class, ties return None (the sample is skipped).
    """
    p1 = row["descriptors"]["path1"][attribute]
    p2 = row["descriptors"]["path2"][attribute]
    if p1 < p2:
        return 0
    if p2 < p1:
        return 1
    return 2 if "Tie" in ATTRIBUTES[attribute]["classes"] else None


def group_split(rows, val_fraction=0.2, seed=SEED):
    """Split rows into (train, val) so that identical descriptor combinations never straddle.

    Mirrors the notebook: group IDs are assigned in sorted order of the descriptor tuple,
    then sklearn's GroupShuffleSplit picks validation groups.
    """
    from sklearn.model_selection import GroupShuffleSplit

    def key(r):
        d1, d2 = r["descriptors"]["path1"], r["descriptors"]["path2"]
        return (d1["Path length"], d2["Path length"], d1["Sharp turns"], d2["Sharp turns"])

    keys = [key(r) for r in rows]
    group_of = {k: i for i, k in enumerate(sorted(set(keys)))}
    groups = [group_of[k] for k in keys]
    splitter = GroupShuffleSplit(n_splits=1, test_size=val_fraction, random_state=seed)
    train_idx, val_idx = next(splitter.split(rows, groups=groups))
    train, val = [rows[i] for i in train_idx], [rows[i] for i in val_idx]
    assert not {key(r) for r in train} & {key(r) for r in val}, "group leak between splits"
    return train, val


def oracle_attributes(row, swap=False, attributes=tuple(ATTRIBUTES)):
    """Ground-truth attributes in the same format as learned predictions (confidence 1.0).

    Used only for the clearly named oracle (upper-bound) condition.
    """
    out = {}
    for attr in attributes:
        idx = attribute_label(row, attr)
        name = "Tie" if idx in (2, None) else ("Path 1", "Path 2")[idx]
        out[attr] = {"lower": _flip_name(name) if swap else name, "confidence": 1.0,
                     "source": "oracle"}
    return out


def _flip_name(name):
    return {"Path 1": "Path 2", "Path 2": "Path 1"}.get(name, name)


# --------------------------------------------------------------------------
# CLIP features
# --------------------------------------------------------------------------

def _device():
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class ClipPairEncoder:
    """Frozen CLIP image encoder for the two halves of a side-by-side path image."""

    def __init__(self, model_name=CLIP_MODEL, device=None):
        import torch
        from transformers import CLIPModel, CLIPProcessor

        self.torch = torch
        self.model_name = model_name
        self.device = device or _device()
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model = CLIPModel.from_pretrained(model_name).to(self.device).eval()
        for p in self.model.parameters():
            p.requires_grad = False

    def encode_images(self, images):
        """L2-normalized CLIP embeddings for a list of PIL images -> [N, 512]."""
        torch = self.torch
        inputs = self.processor(images=images, return_tensors="pt")
        with torch.inference_mode():
            feats = self.model.get_image_features(pixel_values=inputs["pixel_values"].to(self.device))
            if hasattr(feats, "pooler_output"):          # some transformers versions
                feats = feats.pooler_output
            feats = torch.nn.functional.normalize(feats, p=2, dim=-1)
        return feats.float().cpu()

    def encode_rows(self, rows, data_root, batch_size=32, progress=True):
        """Per-half embeddings for manifest rows -> [N, 2, 512] (Path 1, Path 2 as stored)."""
        torch = self.torch
        out = []
        for i in range(0, len(rows), batch_size):
            halves = []
            for row in rows[i:i + batch_size]:
                img = load_pair_image(row, data_root)
                w, h = img.size
                halves += [img.crop((0, 0, w // 2, h)), img.crop((w // 2, 0, w, h))]
            feats = self.encode_images(halves)             # [2B, D], alternating Path 1 / Path 2
            out.append(feats.view(-1, 2, feats.shape[-1]))  # [B, 2, D]
            if progress:
                print(f"  CLIP features {min(i + batch_size, len(rows))}/{len(rows)}", end="\r")
        if progress:
            print()
        return torch.cat(out)


def pair_features(halves, swap=False):
    """[N, 2, D] per-half embeddings -> [N, 2D] pair features, optionally with halves swapped."""
    import torch
    first, second = (halves[:, 1], halves[:, 0]) if swap else (halves[:, 0], halves[:, 1])
    return torch.cat([first, second], dim=1)


# --------------------------------------------------------------------------
# Heads and training
# --------------------------------------------------------------------------

def make_head(kind, n_classes, input_dim=1024, hidden_dim=128, dropout=0.2):
    import torch.nn as nn
    if kind == "linear":
        return nn.Linear(input_dim, n_classes)
    if kind == "mlp":
        return nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout),
                             nn.Linear(hidden_dim, n_classes))
    raise ValueError(f"Unknown head kind: {kind!r}")


def class_weights_for(y, n_classes):
    """Inverse-frequency weights normalized to mean 1 (as in the notebook)."""
    import torch
    counts = torch.bincount(y, minlength=n_classes).float()
    w = counts.sum() / (n_classes * counts.clamp(min=1))
    return w / w.mean()


def train_head(X_train, y_train, X_val, y_val, kind, n_classes, use_class_weights=False,
               lr=1e-3, weight_decay=1e-3, max_epochs=200, patience=20, seed=SEED):
    """Full-batch training with early stopping on validation loss. Returns (head, info)."""
    import torch
    import torch.nn as nn

    torch.manual_seed(seed)
    head = make_head(kind, n_classes, input_dim=X_train.shape[1])
    weights = class_weights_for(y_train, n_classes) if use_class_weights else None
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.Adam(head.parameters(), lr=lr, weight_decay=weight_decay)

    best_loss, best_state, best_epoch, wait, history = float("inf"), None, 0, 0, []
    for epoch in range(1, max_epochs + 1):
        head.train()
        optimizer.zero_grad()
        loss = criterion(head(X_train), y_train)
        loss.backward()
        optimizer.step()

        head.eval()
        with torch.no_grad():
            val_logits = head(X_val)
            val_loss = criterion(val_logits, y_val).item()
            train_acc = (head(X_train).argmax(1) == y_train).float().mean().item()
            val_acc = (val_logits.argmax(1) == y_val).float().mean().item()
        history.append({"epoch": epoch, "train_loss": loss.item(), "val_loss": val_loss,
                        "train_acc": train_acc, "val_acc": val_acc})
        if val_loss < best_loss - 1e-4:
            best_loss, best_state, best_epoch, wait = val_loss, copy.deepcopy(head.state_dict()), epoch, 0
        else:
            wait += 1
            if wait >= patience:
                break

    head.load_state_dict(best_state)
    head.eval()
    return head, {"best_epoch": best_epoch, "best_val_loss": best_loss, "history": history,
                  **head_metrics(head, X_train, y_train, X_val, y_val, n_classes)}


def head_metrics(head, X_train, y_train, X_val, y_val, n_classes):
    import torch
    from sklearn.metrics import f1_score

    with torch.no_grad():
        train_pred = head(X_train).argmax(1)
        val_pred = head(X_val).argmax(1)
    majority = torch.bincount(y_train, minlength=n_classes).argmax().item()
    return {
        "train_acc": (train_pred == y_train).float().mean().item(),
        "val_acc": (val_pred == y_val).float().mean().item(),
        "val_macro_f1": float(f1_score(y_val.numpy(), val_pred.numpy(),
                                       labels=list(range(n_classes)), average="macro",
                                       zero_division=0)),
        "val_majority_baseline": (y_val == majority).float().mean().item(),
        "n_train": len(y_train), "n_val": len(y_val),
    }


def train_grounding(train_rows, data_root, encoder, kinds=("linear", "mlp"),
                    attributes=tuple(ATTRIBUTES), val_fraction=0.2, seed=SEED):
    """Train every (attribute, head kind) on the training rows.

    Returns a bundle with the selected head per attribute and the full comparison.
    """
    import torch

    train, val = group_split(train_rows, val_fraction, seed)
    print(f"Grounding split: {len(train)} train / {len(val)} validation (grouped)")
    X_train = pair_features(encoder.encode_rows(train, data_root))
    X_val = pair_features(encoder.encode_rows(val, data_root))

    bundle = {"clip_model": encoder.model_name, "seed": seed, "heads": {}, "comparison": []}
    for attr in attributes:
        spec = ATTRIBUTES[attr]
        n_classes = len(spec["classes"])
        keep_tr = [i for i, r in enumerate(train) if attribute_label(r, attr) is not None]
        keep_va = [i for i, r in enumerate(val) if attribute_label(r, attr) is not None]
        y_tr = torch.tensor([attribute_label(train[i], attr) for i in keep_tr])
        y_va = torch.tensor([attribute_label(val[i], attr) for i in keep_va])
        candidates = {}
        for kind in kinds:
            head, info = train_head(X_train[keep_tr], y_tr, X_val[keep_va], y_va, kind, n_classes,
                                    use_class_weights=spec["class_weights"], seed=seed)
            candidates[kind] = (head, info)
            bundle["comparison"].append({"attribute": attr, "head": kind,
                                         **{k: v for k, v in info.items() if k != "history"}})
            print(f"  {attr:12s} {kind:6s}  val acc {info['val_acc']:.3f}  "
                  f"macro-F1 {info['val_macro_f1']:.3f}  "
                  f"(majority {info['val_majority_baseline']:.3f}, best epoch {info['best_epoch']})")
        best_kind = max(candidates, key=lambda k: candidates[k][1]["val_acc"])
        head, info = candidates[best_kind]
        bundle["heads"][attr] = {"kind": best_kind, "classes": list(spec["classes"]),
                                 "head": head, "info": info}
        print(f"  -> {attr}: using {best_kind}")
    return bundle


def _input_dim(head):
    first = head if not hasattr(head, "__getitem__") else head[0]
    return first.in_features


def save_grounding(bundle, path):
    import torch
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "clip_model": bundle["clip_model"], "seed": bundle["seed"],
        "comparison": bundle["comparison"],
        "heads": {attr: {"kind": h["kind"], "classes": h["classes"],
                         "input_dim": _input_dim(h["head"]),
                         "state_dict": h["head"].state_dict(),
                         "info": {k: v for k, v in h["info"].items() if k != "history"}}
                  for attr, h in bundle["heads"].items()},
    }, path)


def load_grounding(path):
    import torch
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    for attr, h in ckpt["heads"].items():
        head = make_head(h["kind"], len(h["classes"]), input_dim=h.get("input_dim", 1024))
        head.load_state_dict(h["state_dict"])
        h["head"] = head.eval()
    return ckpt


# --------------------------------------------------------------------------
# Prediction
# --------------------------------------------------------------------------

def predict_attributes(rows, data_root, bundle, encoder, swaps=(False, True)):
    """Predict attributes for each row, for the image as stored and/or with halves swapped.

    Returns records: {"sample_id", "swapped", "attributes": {attr: {"lower", "confidence",
    "head", "source"}}}. "lower" is "Path 1", "Path 2", or "Tie" relative to the image as shown.
    """
    import torch

    halves = encoder.encode_rows(rows, data_root)
    records = []
    for swap in swaps:
        X = pair_features(halves, swap=swap)
        per_attr = {}
        for attr, h in bundle["heads"].items():
            with torch.no_grad():
                probs = torch.softmax(h["head"](X), dim=1)
            conf, idx = probs.max(dim=1)
            per_attr[attr] = [(h["classes"][i], float(c), h["kind"]) for i, c in zip(idx.tolist(), conf)]
        for n, row in enumerate(rows):
            records.append({
                "sample_id": row["sample_id"], "swapped": swap,
                "attributes": {attr: {"lower": v[n][0], "confidence": round(v[n][1], 4),
                                      "head": v[n][2], "source": "clip"}
                               for attr, v in per_attr.items()},
            })
    return records


def load_attribute_file(path):
    """Read predicted attributes into {(sample_id, swapped): attributes}."""
    return {(r["sample_id"], r["swapped"]): r["attributes"] for r in read_jsonl(path)}


# --------------------------------------------------------------------------
# Command line
# --------------------------------------------------------------------------

def main(argv=None):
    parser = argparse.ArgumentParser(description="CLIP + linear/MLP grounding for PathEval")
    sub = parser.add_subparsers(dest="command", required=True)

    t = sub.add_parser("train", help="train heads on the training split")
    t.add_argument("--manifest", required=True)
    t.add_argument("--data-root", required=True)
    t.add_argument("--out", required=True, help="checkpoint path, e.g. models/grounding.pt")
    t.add_argument("--n-train", type=int, default=500,
                   help="use the first N training records (notebook used 500)")
    t.add_argument("--kinds", nargs="+", default=["linear", "mlp"])

    p = sub.add_parser("predict", help="predict attributes for a manifest")
    p.add_argument("--manifest", required=True)
    p.add_argument("--data-root", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--n-per-scenario", type=int, default=None)
    p.add_argument("--seed", type=int, default=0, help="subset seed (match the prompting runs)")

    args = parser.parse_args(argv)
    encoder = ClipPairEncoder()
    if args.command == "train":
        rows = load_manifest(args.manifest)[:args.n_train]
        bundle = train_grounding(rows, args.data_root, encoder, kinds=tuple(args.kinds))
        save_grounding(bundle, args.out)
        with open(f"{args.out}.comparison.json", "w") as f:
            json.dump(bundle["comparison"], f, indent=2)
        print(f"Saved {args.out}")
    else:
        rows = load_manifest(args.manifest)
        if args.n_per_scenario:
            rows = stratified_subset(rows, args.n_per_scenario, args.seed)
        records = predict_attributes(rows, args.data_root, load_grounding(args.model), encoder)
        write_jsonl(records, args.out)
        print(f"Wrote {len(records)} attribute records to {args.out}")


if __name__ == "__main__":
    main()
