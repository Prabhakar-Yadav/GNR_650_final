#!/bin/bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_DIR"

STABLE_MODEL_ID="Qwen/Qwen2.5-VL-7B-Instruct"
REQUESTED_MODEL_ID="${VLM_MODEL_ID:-$STABLE_MODEL_ID}"
if [[ "${REQUESTED_MODEL_ID^^}" == *AWQ* && "${ALLOW_UNSTABLE_AWQ:-0}" != "1" ]]; then
    echo "WARNING: ${REQUESTED_MODEL_ID} is an AWQ model and is disabled by default."
    echo "The tested AutoAWQ/Triton stack can crash during generation."
    echo "Using stable ${STABLE_MODEL_ID} instead."
    echo "Set ALLOW_UNSTABLE_AWQ=1 only if you intentionally want to test AWQ."
    MODEL_ID="$STABLE_MODEL_ID"
else
    MODEL_ID="$REQUESTED_MODEL_ID"
fi
MODEL_DIR="${VLM_MODEL_DIR:-./model_weights/${MODEL_ID##*/}}"
PYTHON_BIN="${PYTHON_BIN:-python}"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    if command -v python3 >/dev/null 2>&1; then
        PYTHON_BIN="python3"
    else
        echo "ERROR: python/python3 not found. Please run on Python 3.11." >&2
        exit 1
    fi
fi

echo "Using Python: $("$PYTHON_BIN" --version)"
echo "Repository: ${REPO_DIR}"

"$PYTHON_BIN" - <<'PYEOF'
import sys

if sys.version_info[:2] != (3, 11):
    raise SystemExit(
        f"Python 3.11 is required for this submission; found "
        f"{sys.version_info.major}.{sys.version_info.minor}."
    )
PYEOF

"$PYTHON_BIN" -m pip install --upgrade pip

# CUDA 12.4 wheels are compatible with the CUDA 12.6 driver stack on the L40s.
"$PYTHON_BIN" -m pip install \
    torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 \
    --index-url https://download.pytorch.org/whl/cu124

# Install the main inference stack first.
"$PYTHON_BIN" -m pip install \
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

if [[ "${MODEL_ID^^}" == *AWQ* ]]; then
    # AutoAWQ can change the Transformers version, so force Transformers back to
    # the Qwen2.5-VL-compatible build after AutoAWQ is installed.
    "$PYTHON_BIN" -m pip install autoawq==0.2.9
    "$PYTHON_BIN" -m pip install --upgrade --no-deps transformers==4.51.3
fi

echo "Downloading VLM weights: ${MODEL_ID}"
export MODEL_ID MODEL_DIR
"$PYTHON_BIN" - <<'PYEOF'
import os
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id=os.environ["MODEL_ID"],
    local_dir=os.environ["MODEL_DIR"],
    ignore_patterns=["*.msgpack", "*.h5", "flax_model*", "*.ot"],
)
print(f"{os.environ['MODEL_ID']} weights are ready.")
PYEOF

echo "Downloading EasyOCR English detection/recognition assets"
"$PYTHON_BIN" - <<'PYEOF'
import easyocr

easyocr.Reader(["en"], gpu=False, verbose=False)
print("EasyOCR assets are ready.")
PYEOF

echo ""
echo "Setup complete."
echo "Run inference with: ${PYTHON_BIN} inference.py --test_dir <absolute_path_to_test_dir>"
