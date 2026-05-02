# Modal Offline Bot Test

This branch is for Modal-based bot testing with internet disabled during inference.

## Recommended Stable 32-48GB GPU Run

Use Qwen2.5-VL-7B-Instruct for the final bot-style test. This avoids the
AutoAWQ/Triton generation crash seen with Qwen2.5-VL-32B-AWQ.

```bash
pip install modal
modal setup

git clone -b codex/modal-robust-test https://github.com/Prabhakar-Yadav/GNR_650_final.git
cd GNR_650_final

modal volume create gnr-test-data || true
modal volume put gnr-test-data /absolute/path/to/test_dir /test_dir -f

MODAL_GPU=A100-40GB \
VLM_MODEL_ID=Qwen/Qwen2.5-VL-7B-Instruct \
QWEN_MAX_PIXELS=602112 \
modal run modal_bot.py --test-dir /data/test_dir --out submission_modal.csv
```

`/absolute/path/to/test_dir` must contain:

```text
test.csv
sample_submission.csv
patches/
```

The Modal image build downloads dependencies, VLM weights, and EasyOCR assets. The remote function then runs with `block_network=True` plus HuggingFace offline environment variables.

## AWQ Stress Test Only

AWQ is disabled by default because the tested AutoAWQ/Triton stack can crash
during generation. Use this only for experiments, not the final submission.

```bash
ALLOW_UNSTABLE_AWQ=1 \
MODAL_GPU=A100-80GB \
VLM_MODEL_ID=Qwen/Qwen2.5-VL-32B-Instruct-AWQ \
QWEN_MAX_PIXELS=602112 \
modal run modal_bot.py --test-dir /data/test_dir --out submission_modal_awq.csv
```

## Windows PowerShell

```powershell
pip install modal
modal setup

git clone -b codex/modal-robust-test https://github.com/Prabhakar-Yadav/GNR_650_final.git
cd GNR_650_final

modal volume create gnr-test-data
modal volume put gnr-test-data C:\absolute\path\to\test_dir /test_dir -f

$env:MODAL_GPU="A100-40GB"
$env:VLM_MODEL_ID="Qwen/Qwen2.5-VL-7B-Instruct"
$env:QWEN_MAX_PIXELS="602112"
modal run modal_bot.py --test-dir /data/test_dir --out submission_modal.csv
```
