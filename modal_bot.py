import os
import pathlib
import subprocess

import modal


MODEL_ID = os.environ.get("VLM_MODEL_ID", "Qwen/Qwen2.5-VL-32B-Instruct-AWQ")
GPU = os.environ.get("MODAL_GPU", "A100-40GB")
TEST_VOLUME_NAME = os.environ.get("MODAL_TEST_VOLUME", "gnr-test-data")

app = modal.App("gnr-final-offline-bot")
test_volume = modal.Volume.from_name(TEST_VOLUME_NAME, create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("bash", "git")
    .add_local_dir(".", remote_path="/root/project")
    .workdir("/root/project")
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
            "QWEN_MAX_PIXELS": str(1280 * 28 * 28),
        }
    )
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
            "QWEN_MAX_PIXELS": str(1280 * 28 * 28),
        }
    )
    result = subprocess.run(
        ["python", "inference.py", "--test_dir", test_dir],
        cwd="/root/project",
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=55 * 60,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stdout[-8000:])

    submission = pathlib.Path("/root/project/submission.csv").read_text()
    return {"submission": submission, "log_tail": result.stdout[-8000:]}


@app.local_entrypoint()
def main(test_dir="/data/test_dir", out="submission_modal.csv"):
    result = run_bot.remote(test_dir)
    pathlib.Path(out).write_text(result["submission"])
    print(result["log_tail"])
    print(f"Wrote {out}")
