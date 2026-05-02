import os
import pathlib
import subprocess

import modal


STABLE_MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"
REQUESTED_MODEL_ID = os.environ.get("VLM_MODEL_ID", STABLE_MODEL_ID)
ALLOW_UNSTABLE_AWQ = os.environ.get("ALLOW_UNSTABLE_AWQ", "0") == "1"
if "AWQ" in REQUESTED_MODEL_ID.upper() and not ALLOW_UNSTABLE_AWQ:
    print(
        f"Requested {REQUESTED_MODEL_ID}, but AWQ is disabled because the "
        f"current AutoAWQ/Triton stack crashes during generation. "
        f"Using stable {STABLE_MODEL_ID}. Set ALLOW_UNSTABLE_AWQ=1 to force AWQ."
    )
    MODEL_ID = STABLE_MODEL_ID
else:
    MODEL_ID = REQUESTED_MODEL_ID
GPU = os.environ.get("MODAL_GPU", "A100-40GB")
TEST_VOLUME_NAME = os.environ.get("MODAL_TEST_VOLUME", "gnr-test-data")
QWEN_MAX_PIXELS = os.environ.get("QWEN_MAX_PIXELS", str(768 * 28 * 28))
PROJECT_DIR = "/root/project"

app = modal.App("gnr-final-offline-bot")
test_volume = modal.Volume.from_name(TEST_VOLUME_NAME, create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("bash", "libglib2.0-0")
    .add_local_file("setup.bash", f"{PROJECT_DIR}/setup.bash", copy=True)
    .workdir(PROJECT_DIR)
    .env(
        {
            "VLM_MODEL_ID": MODEL_ID,
            "TOKENIZERS_PARALLELISM": "false",
        }
    )
    .run_commands("bash setup.bash")
    .env(
        {
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "HF_DATASETS_OFFLINE": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "QWEN_MAX_PIXELS": QWEN_MAX_PIXELS,
        }
    )
    .add_local_file("requirements.txt", f"{PROJECT_DIR}/requirements.txt")
    .add_local_file("inference.py", f"{PROJECT_DIR}/inference.py")
    .add_local_file("sample_submission.csv", f"{PROJECT_DIR}/sample_submission.csv")
)


@app.function(
    image=image,
    gpu=GPU,
    volumes={"/data": test_volume.read_only()},
    timeout=60 * 60,
    block_network=True,
)
def run_bot(test_dir="/data/test_dir"):
    env = os.environ.copy()
    env.update(
        {
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "HF_DATASETS_OFFLINE": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "QWEN_MAX_PIXELS": QWEN_MAX_PIXELS,
        }
    )
    test_path = pathlib.Path(test_dir)
    if not (test_path / "test.csv").exists() or not (test_path / "patches").is_dir():
        raise FileNotFoundError(
            f"{test_dir} must contain test.csv and patches/. "
            f"Upload with: modal volume put {TEST_VOLUME_NAME} <local_test_dir> /test_dir -f"
        )

    result = subprocess.run(
        ["python", "inference.py", "--test_dir", test_dir],
        cwd=PROJECT_DIR,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=55 * 60,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stdout[-8000:])

    submission_path = pathlib.Path(PROJECT_DIR) / "submission.csv"
    submission = submission_path.read_text()
    return {
        "submission": submission,
        "log_tail": result.stdout[-8000:],
        "model_id": MODEL_ID,
        "gpu": GPU,
        "qwen_max_pixels": QWEN_MAX_PIXELS,
    }


@app.local_entrypoint()
def main(test_dir="/data/test_dir", out="submission_modal.csv"):
    result = run_bot.remote(test_dir)
    pathlib.Path(out).write_text(result["submission"])
    print(f"Model: {result['model_id']}")
    print(f"GPU: {result['gpu']}")
    print(f"QWEN_MAX_PIXELS: {result['qwen_max_pixels']}")
    print(result["log_tail"])
    print(f"Wrote {out}")
