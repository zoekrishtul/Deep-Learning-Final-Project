"""PathEval data utilities: manifest adapter, subsets, and image loading.

Converts the released PathEval records into the team manifest contract
(docs/README.md) and loads the side-by-side path images.

Release quirks handled here (verified on the 2D test split):
  * Images live in subfolders named after the first digit of the image ID
    (``combined_images/2/2383_2D.png``), but at least one record omits that
    subfolder (``combined_images/2_2D.png``). ``resolve_image_path`` fixes this.
  * Each image is 960x480: Path 1 on the left half, Path 2 on the right half.
  * ``combined_images_2`` is a byte-identical copy of ``combined_images`` for
    all 1050 test images, so only ``combined_images`` is used.
"""

import json
import random
from collections import defaultdict
from pathlib import Path


# --------------------------------------------------------------------------
# Manifest I/O
# --------------------------------------------------------------------------

def read_jsonl(path):
    """Read a JSONL file into a list of dicts."""
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(rows, path):
    """Write a list of dicts to a JSONL file (one JSON object per line)."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def load_manifest(path):
    """Load a manifest and check the fields every module relies on."""
    rows = read_jsonl(path)
    ids = [r["sample_id"] for r in rows]
    if len(ids) != len(set(ids)):
        raise ValueError(f"Duplicate sample_id values in {path}")
    for r in rows:
        if r["label"] not in (1, 2):
            raise ValueError(f"Bad label {r['label']!r} for sample {r['sample_id']}")
    return rows


# --------------------------------------------------------------------------
# Adapting the released data
# --------------------------------------------------------------------------

def resolve_image_path(image, repo_files, dim="2D"):
    """Map an image path from the released JSON to its real repo location.

    Tries the path as written, then with the first-digit subfolder added.
    Raises FileNotFoundError if neither exists in ``repo_files``.
    """
    folder, name = image.rsplit("/", 1)
    for candidate in (f"{dim}/{image}", f"{dim}/{folder}/{name[0]}/{name}"):
        if candidate in repo_files:
            return candidate
    raise FileNotFoundError(f"No file found for {image!r}")


def parse_label(text):
    """'Path 1' -> 1, 'Path 2' -> 2."""
    label = int(text.strip().split()[-1])
    if label not in (1, 2):
        raise ValueError(f"Unexpected label {text!r}")
    return label


def to_manifest_row(sample_id, image_path, scenario, label, path1_desc, path2_desc,
                    split="test", dim="2D"):
    """Build one manifest row in the team contract format.

    ``image_paths`` holds a single side-by-side image (Path 1 left, Path 2 right),
    relative to the data root. Ground-truth descriptors are kept as evaluation /
    oracle metadata only; visual-only prompting conditions never read them.
    """
    return {
        "sample_id": str(sample_id),
        "split": split,
        "dimension": dim,
        "image_paths": [image_path],
        "scenario": scenario,
        "label": label,
        "descriptors": {"path1": path1_desc, "path2": path2_desc},
    }


def manifest_from_release(records, repo_files, split="test", dim="2D"):
    """Convert raw HuggingFace PathEval records into manifest rows."""
    return [
        to_manifest_row(
            r["id"], resolve_image_path(r["image"], repo_files, dim), r["scenario"],
            parse_label(r["ground truth"]), r["path 1 descriptors"], r["path 2 descriptors"],
            split, dim)
        for r in records
    ]


def manifest_from_clean(records, split="test", dim="2D"):
    """Convert the earlier ``test_2D_clean.json`` (data-setup notebook v1) to manifest rows."""
    return [
        to_manifest_row(
            r["id"], r["image"], r["scenario"], parse_label(r["label"]),
            r["path1_descriptors"], r["path2_descriptors"], split, dim)
        for r in records
    ]


RELEASE_JSON = {"train": "{dim}/train_set_{dim}.json", "test": "{dim}/test_dataset_{dim}.json"}


def fetch_patheval(split, local_dir, dim="2D", limit=None, repo="maghzal/PathEval"):
    """Download a PathEval split (optionally only the first ``limit`` records) and its images.

    Writes ``<local_dir>/<split>_<dim>.jsonl`` (team manifest, image paths relative to
    ``local_dir``) and returns its path. Already-downloaded images are reused.
    """
    from huggingface_hub import HfApi, hf_hub_download, snapshot_download

    repo_files = set(HfApi().list_repo_files(repo, repo_type="dataset"))
    with open(hf_hub_download(repo, RELEASE_JSON[split].format(dim=dim), repo_type="dataset")) as f:
        records = json.load(f)
    records = records[:limit] if limit else records
    rows = manifest_from_release(records, repo_files, split, dim)
    images = [r["image_paths"][0] for r in rows]
    snapshot_download(repo, repo_type="dataset", allow_patterns=images, local_dir=local_dir)
    missing = [p for p in images if not (Path(local_dir) / p).exists()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} images failed to download, e.g. {missing[:3]}")
    out = Path(local_dir) / f"{split}_{dim}.jsonl"
    write_jsonl(rows, out)
    print(f"{split}: {len(rows)} records (first id {rows[0]['sample_id']}) -> {out}")
    return out


# --------------------------------------------------------------------------
# Subsets
# --------------------------------------------------------------------------

def stratified_subset(rows, n_per_scenario, seed=0):
    """Pick the same ``n_per_scenario`` examples from every scenario.

    Deterministic for a given manifest order and seed, so every prompting
    condition is evaluated on exactly the same examples. Scenarios are visited
    in order of first appearance.
    """
    by_scenario = defaultdict(list)
    for r in rows:
        by_scenario[r["scenario"]].append(r)
    rng = random.Random(seed)
    subset = []
    for examples in by_scenario.values():
        subset.extend(rng.sample(examples, min(n_per_scenario, len(examples))))
    return subset


# --------------------------------------------------------------------------
# Images
# --------------------------------------------------------------------------

def swap_halves(image):
    """Return a copy of a side-by-side image with the left and right halves swapped."""
    from PIL import Image

    w, h = image.size
    left = image.crop((0, 0, w // 2, h))
    right = image.crop((w // 2, 0, w, h))
    out = Image.new(image.mode, (w, h))
    out.paste(right, (0, 0))
    out.paste(left, (w // 2, 0))
    return out


def load_pair_image(row, data_root, swap=False):
    """Load a sample's side-by-side image as RGB, optionally with halves swapped."""
    from PIL import Image

    with Image.open(Path(data_root) / row["image_paths"][0]) as img:
        img = img.convert("RGB")
    return swap_halves(img) if swap else img
