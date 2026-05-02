# Grading Bot Command Sequence

These are the commands to run from a fresh Linux machine in the same style as the evaluator.

## Full Remote Test

```bash
git clone https://github.com/Prabhakar-Yadav/GNR_650_final.git
cd GNR_650_final
bash setup.bash
python inference.py --test_dir /absolute/path/to/test_dir
```

The test directory must contain:

```text
test_dir/
  patches/
    patch_0.png
    patch_1.png
    ...
  test.csv
```

`inference.py` writes `submission.csv` in the current directory.

## Accuracy Check When Ground Truth Exists

Use this only for the visible/local validation set where `test.csv` includes `correct_answer`.

```bash
python evaluate.py --predictions submission.csv --ground_truth /absolute/path/to/test_dir/test.csv
```

## Fast CPU/No-Model Smoke Test

This checks patch loading, stitching, OCR, CSV writing, and valid answer formatting without loading the VLM.

```bash
SKIP_VLM=1 python inference.py --test_dir /absolute/path/to/test_dir
python - <<'PY'
import pandas as pd
s = pd.read_csv("submission.csv")
assert list(s.columns) == ["id", "question_num", "option"]
assert s["option"].isin([1, 2, 3, 4, 5]).all()
print("submission.csv format is valid:", len(s), "rows")
PY
```

## What setup.bash Downloads

- Dependencies installed into the current Python 3.11 environment
- PyTorch 2.6 CUDA 12.4 wheels, compatible with CUDA 12.6 drivers
- Transformers 4.51.3, AutoAWQ 0.2.9, qwen-vl-utils 0.0.8
- Qwen2.5-VL-72B-Instruct-AWQ weights
- EasyOCR English detection/recognition assets

After setup, inference is local/offline: do not rely on internet during `python inference.py`.
