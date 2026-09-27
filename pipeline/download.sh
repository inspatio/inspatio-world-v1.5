#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p "$SCRIPT_DIR/checkpoints/InSpatio-World-1.3B" "$SCRIPT_DIR/checkpoints/depth"
cd "$SCRIPT_DIR/checkpoints"

# --- Download the default InSpatio-World 1.5 checkpoint ---
if [ ! -f InSpatio-World-1.3B/InSpatio-World-1.5-1.3B.safetensors ]; then
    python - <<'PY'
from pathlib import Path
import os
from huggingface_hub import hf_hub_download, list_repo_files

repo_id = "inspatio/world-1.5"
model_dir = Path("InSpatio-World-1.3B")
target = model_dir / "InSpatio-World-1.5-1.3B.safetensors"
files = [name for name in list_repo_files(repo_id) if name.endswith(".safetensors")]
preferred = [name for name in files if Path(name).name in {
    "InSpatio-World-1.5-1.3B.safetensors",
    "InSpatio-World-1.5.safetensors",
}]
if len(preferred) == 1:
    filename = preferred[0]
elif len(files) == 1:
    filename = files[0]
else:
    raise RuntimeError(f"Cannot identify the 1.5 checkpoint in {repo_id}: {files}")
downloaded = Path(hf_hub_download(repo_id=repo_id, filename=filename, local_dir=model_dir))
if downloaded != target:
    os.replace(downloaded, target)
print(f"Saved {repo_id}/{filename} to {target}")
PY
fi

# --- Download the Wan-AI series ---
if [ ! -e Wan2.1-T2V-1.3B ]; then
    GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B
    cd Wan2.1-T2V-1.3B && git lfs pull && cd ..
fi

# --- Download depth estimator config and weights ---
if [ ! -f depth/config.json ]; then
    python -c 'from huggingface_hub import hf_hub_download; hf_hub_download(repo_id="depth-anything/DA3NESTED-GIANT-LARGE", filename="config.json", local_dir="depth")'
fi
if [ ! -f depth/model.safetensors ]; then
    python -c 'from huggingface_hub import hf_hub_download; hf_hub_download(repo_id="depth-anything/DA3NESTED-GIANT-LARGE", filename="model.safetensors", local_dir="depth")'
fi

cd ..

echo "All models have been downloaded and organized into the checkpoints/ directory."
