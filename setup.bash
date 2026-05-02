#!/bin/bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_DIR"

ENV_NAME="gnr_project_env"
MODEL_ID="Qwen/Qwen2.5-VL-72B-Instruct-AWQ"

echo "Preparing ${ENV_NAME} in ${REPO_DIR}"

# Re-create the environment so repeated grading runs start cleanly.
conda remove --name "$ENV_NAME" --all -y 2>/dev/null || true
conda create -n "$ENV_NAME" python=3.11 -y

conda run -n "$ENV_NAME" python -m pip install --upgrade pip

# CUDA 12.4 wheels are compatible with the CUDA 12.6 driver stack on the L40s.
conda run -n "$ENV_NAME" python -m pip install \
    torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 \
    --index-url https://download.pytorch.org/whl/cu124

# Install the main inference stack first.
conda run -n "$ENV_NAME" python -m pip install \
    transformers==4.51.3 \
    accelerate==1.6.0 \
    huggingface_hub \
    qwen-vl-utils==0.0.8 \
    pillow \
    opencv-python-headless \
    pandas \
    numpy \
    easyocr \
    bitsandbytes

# AutoAWQ can change the Transformers version, so force Transformers back to
# the Qwen2.5-VL-compatible build after AutoAWQ is installed.
conda run -n "$ENV_NAME" python -m pip install autoawq==0.2.9
conda run -n "$ENV_NAME" python -m pip install --upgrade --no-deps transformers==4.51.3

echo "Downloading VLM weights: ${MODEL_ID}"
conda run -n "$ENV_NAME" python - <<'PYEOF'
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="Qwen/Qwen2.5-VL-72B-Instruct-AWQ",
    local_dir="./model_weights/Qwen2.5-VL-72B-Instruct-AWQ",
    ignore_patterns=["*.msgpack", "*.h5", "flax_model*", "*.ot"],
)
print("Qwen2.5-VL-72B-AWQ weights are ready.")
PYEOF

echo "Downloading EasyOCR English detection/recognition assets"
conda run -n "$ENV_NAME" python - <<'PYEOF'
import easyocr

easyocr.Reader(["en"], gpu=False, verbose=False)
print("EasyOCR assets are ready.")
PYEOF

echo ""
echo "Setup complete."
echo "Activate with: conda activate ${ENV_NAME}"
echo "Run inference: python inference.py --test_dir <absolute_path_to_test_dir>"
