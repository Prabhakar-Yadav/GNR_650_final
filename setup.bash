#!/bin/bash
set -e

# ─── 1. Clone the repository ──────────────────────────────────────────────────
# IMPORTANT: Replace with your actual public GitHub repo URL before submitting
REPO_URL="https://github.com/Prabhakar-Yadav/GNR_650_final.git"
REPO_DIR="gnr_final_project_repo"

if [ ! -d "$REPO_DIR" ]; then
    git clone "$REPO_URL" "$REPO_DIR"
fi

# Copy inference.py to the working directory so the grader can run:
#   python inference.py --test_dir <path>
# from this same directory (where setup.bash lives).
cp "$REPO_DIR/inference.py" ./inference.py

# ─── 2. Create conda environment (Python 3.11, name: gnr_project_env) ─────────
# Remove if already exists (grading system runs conda remove at end, but handle re-runs)
conda remove --name gnr_project_env --all -y 2>/dev/null || true
conda create -n gnr_project_env python=3.11 -y

# ─── 3. Install all dependencies ──────────────────────────────────────────────
conda run -n gnr_project_env pip install --upgrade pip

# PyTorch with CUDA 12.4 wheels (compatible with CUDA 12.6 on the L40s)
conda run -n gnr_project_env pip install \
    torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 \
    --index-url https://download.pytorch.org/whl/cu124

# VLM inference stack + utilities
conda run -n gnr_project_env pip install \
    transformers==4.46.3 \
    accelerate==1.1.1 \
    huggingface_hub \
    qwen-vl-utils \
    pillow \
    opencv-python-headless \
    pandas \
    numpy

# ─── 4. Download model weights (internet available here) ──────────────────────
# Qwen2-VL-72B-Instruct-AWQ: 4-bit quantized, ~36GB VRAM, fits in L40s 48GB
# Much better at reading small map text than the 7B model.
conda run -n gnr_project_env pip install autoawq

conda run -n gnr_project_env python - <<'PYEOF'
from huggingface_hub import snapshot_download
import os

# Primary: 72B AWQ (best accuracy for map text reading)
print("Downloading Qwen2-VL-72B-Instruct-AWQ...")
snapshot_download(
    repo_id="Qwen/Qwen2-VL-72B-Instruct-AWQ",
    local_dir="./model_weights/Qwen2-VL-72B-Instruct-AWQ",
    ignore_patterns=["*.msgpack", "*.h5", "flax_model*", "*.ot"],
)
print("Model weights downloaded successfully.")
PYEOF

echo ""
echo "Setup complete. Environment: gnr_project_env | Python 3.11"
echo "Run: conda activate gnr_project_env && python inference.py --test_dir <path>"
