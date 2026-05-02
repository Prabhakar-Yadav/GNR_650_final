"""
Modal test script — simulates grading bot WITHOUT internet during inference.

Usage:
  python -m modal run modal_test_bot.py

What happens:
  1. Image build  (WITH internet): installs deps, downloads model + EasyOCR assets
  2. Function run  (NO internet) : stitches map, runs OCR+VLM, writes submission.csv
"""

import modal
from pathlib import Path

app = modal.App("gnr-grading-test")

# ── Fixed working directory inside the container ─────────────────────────────
WORKDIR = "/app"
MODEL_DIR = f"{WORKDIR}/model_weights/Qwen2.5-VL-72B-Instruct-AWQ"

# ── Build the image layer-by-layer ───────────────────────────────────────────
image = (
    modal.Image.debian_slim(python_version="3.11")
    # System libs needed by OpenCV / EasyOCR
    .apt_install("libgl1", "libglib2.0-0")
    # PyTorch with CUDA 12.4 wheels
    .pip_install(
        "torch==2.6.0", "torchvision==0.21.0", "torchaudio==2.6.0",
        index_url="https://download.pytorch.org/whl/cu124",
    )
    # Main inference stack
    .pip_install(
        "transformers==4.51.3",
        "accelerate==1.6.0",
        "huggingface_hub",
        "qwen-vl-utils==0.0.8",
        "pillow",
        "opencv-python-headless",
        "pandas",
        "numpy",
        "easyocr",
        "bitsandbytes",
    )
    # AutoAWQ (may downgrade transformers, so pin it back)
    .pip_install("autoawq==0.2.9")
    .pip_install("transformers==4.51.3")
    # Download model weights into a known absolute path
    .run_commands(
        f"mkdir -p {MODEL_DIR} && python -c \""
        f"from huggingface_hub import snapshot_download; "
        f"snapshot_download("
        f"  repo_id='Qwen/Qwen2.5-VL-72B-Instruct-AWQ',"
        f"  local_dir='{MODEL_DIR}',"
        f"  ignore_patterns=['*.msgpack','*.h5','flax_model*','*.ot'],"
        f")\""
    )
    # Pre-download EasyOCR English models
    .run_commands(
        "python -c \"import easyocr; easyocr.Reader(['en'], gpu=False, verbose=False)\""
    )
    # Copy your source code into the image
    .add_local_file("inference.py", f"{WORKDIR}/inference.py")
)

# ── Mount: upload your local test data so the container can read it ──────────
# Change this path to where your patches/ and test.csv live locally:
LOCAL_TEST_DIR = r"c:\Users\PRABHAKAR\Documents\GNR_Final_project"
REMOTE_TEST_DIR = "/test_data"

test_mount = modal.Mount.from_local_dir(
    LOCAL_TEST_DIR,
    remote_path=REMOTE_TEST_DIR,
    condition=lambda path: (
        "patches" in path or path.endswith("test.csv")
    ),
)


@app.function(
    image=image,
    gpu="A100",
    timeout=3600,
    mounts=[test_mount],
)
def run_inference_bot():
    import subprocess, os

    os.chdir(WORKDIR)

    print("\n" + "=" * 80)
    print("GRADING BOT TEST — OFFLINE INFERENCE")
    print("=" * 80)

    # Verify model
    model_dir = Path(MODEL_DIR)
    if model_dir.exists():
        n = len(list(model_dir.iterdir()))
        print(f"✓ Model weights found ({n} files): {model_dir}")
    else:
        print(f"✗ Model weights NOT found: {model_dir}")
        return {"status": "FAILED", "error": "Model not found"}

    # Verify test data
    test_dir = Path(REMOTE_TEST_DIR)
    patches = test_dir / "patches"
    test_csv = test_dir / "test.csv"
    if patches.exists() and test_csv.exists():
        n_patches = len(list(patches.glob("*.png")))
        print(f"✓ Test data found: {n_patches} patches, test.csv present")
    else:
        print(f"✗ Test data NOT found at {test_dir}")
        print(f"  patches/ exists: {patches.exists()}")
        print(f"  test.csv exists: {test_csv.exists()}")
        return {"status": "FAILED", "error": "Test data not found"}

    # Verify EasyOCR cache
    easyocr_cache = Path.home() / ".EasyOCR"
    print(f"{'✓' if easyocr_cache.exists() else '⚠'} EasyOCR cache: {easyocr_cache}")

    print(f"\nRunning: python inference.py --test_dir {REMOTE_TEST_DIR}")
    print("-" * 80)

    result = subprocess.run(
        ["python", "inference.py", "--test_dir", REMOTE_TEST_DIR],
        capture_output=True,
        text=True,
        timeout=1800,
    )

    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr[-1000:])

    print("-" * 80)
    print(f"Exit code: {result.returncode}")

    submission = Path("submission.csv")
    if result.returncode == 0 and submission.exists():
        content = submission.read_text()
        lines = content.strip().splitlines()
        print(f"\n✓ submission.csv: {len(lines) - 1} answers")
        for line in lines[:6]:
            print(f"  {line}")
        return {
            "status": "SUCCESS",
            "rows": len(lines) - 1,
            "preview": lines[:6],
        }

    return {
        "status": "FAILED",
        "exit_code": result.returncode,
        "stdout_tail": result.stdout[-500:],
        "stderr_tail": result.stderr[-500:],
    }


@app.local_entrypoint()
def main():
    import json

    print("\n" + "=" * 80)
    print("MODAL BOT TEST")
    print("=" * 80)
    print(f"Local test data : {LOCAL_TEST_DIR}")
    print(f"Remote mount    : {REMOTE_TEST_DIR}")
    print(f"Model           : Qwen2.5-VL-72B-Instruct-AWQ")
    print("-" * 80)

    result = run_inference_bot.remote()

    print("\n" + "=" * 80)
    print("RESULT")
    print("=" * 80)
    print(json.dumps(result, indent=2))

    if result["status"] == "SUCCESS":
        print("\n✓ BOT TEST PASSED")
    else:
        print("\n✗ BOT TEST FAILED")
