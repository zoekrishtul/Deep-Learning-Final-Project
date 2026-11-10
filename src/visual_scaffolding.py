
"""Zoe: reproducible image-only visual scaffolding.

Split each side-by-side image into separate Path 1 and Path 2 views.
Keep path ordering and don't use labels or ground-truth descriptors.
"""

from pathlib import Path
from PIL import Image


def scaffold(image_paths, output_dir):
    """Write transformed images and return their ordered paths."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = []

    for image_path in image_paths:
        example_id = Path(image_path).stem

        with Image.open(image_path) as image:
            width, height = image.size
            midpoint = width // 2

            path1 = image.crop((0, 0, midpoint, height))
            path2 = image.crop((midpoint, 0, width, height))

            path1_file = output_dir / f"{example_id}_path1.png"
            path2_file = output_dir / f"{example_id}_path2.png"

            path1.save(path1_file)
            path2.save(path2_file)

        results.append({
            "original": str(image_path),
            "path1": str(path1_file),
            "path2": str(path2_file),
        })

    return results
