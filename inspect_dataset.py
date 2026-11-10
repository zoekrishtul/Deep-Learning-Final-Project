
from datasets import load_dataset
from huggingface_hub import hf_hub_download
from src.visual_scaffolding import scaffold
from pathlib import Path

dimension = "3D"  # Change to "2D" for another test
dataset = load_dataset("maghzal/PathEval", dimension)

image_paths = []

for example in dataset["test"].select(range(5)):
    filename = f"{dimension}/{example['image']}"

    downloaded_path = hf_hub_download(
        repo_id="maghzal/PathEval",
        repo_type="dataset",
        filename=filename
    )

    # Keep the dataset ID and dimension in the local filename
    local_path = Path("data") / f"{dimension}_{example['id']}.png"
    local_path.parent.mkdir(parents=True, exist_ok=True)

    if not local_path.exists():
        local_path.write_bytes(Path(downloaded_path).read_bytes())

    image_paths.append(str(local_path))

results = scaffold(image_paths, "results/scaffolded")

print(f"Successfully processed {len(results)} images!")
for result in results:
    print(result["path1"], result["path2"])
