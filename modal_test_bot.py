"""
Modal test — simulates grading bot WITHOUT internet during inference.

Usage:
  python -m modal run modal_test_bot.py
"""

import modal
from pathlib import Path

app = modal.App("gnr-grading-test")

WORKDIR = "/app"
MODEL_DIR = f"{WORKDIR}/model_weights/Qwen2.5-VL-72B-Instruct-AWQ"
TEST_DIR = f"{WORKDIR}/test_data"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libgl1", "libglib2.0-0")
    .pip_install(
        "torch==2.6.0", "torchvision==0.21.0", "torchaudio==2.6.0",
        index_url="https://download.pytorch.org/whl/cu124",
    )
    .pip_install(
        "transformers==4.51.3", "accelerate==1.6.0", "huggingface_hub",
        "qwen-vl-utils==0.0.8", "pillow", "opencv-python-headless",
        "pandas", "numpy", "easyocr", "bitsandbytes",
    )
    .pip_install("autoawq", "autoawq-kernels")
    .pip_install("transformers==4.51.3")
    .run_commands(
        f"mkdir -p {MODEL_DIR} && python -c \""
        f"from huggingface_hub import snapshot_download; "
        f"snapshot_download("
        f"  repo_id='Qwen/Qwen2.5-VL-72B-Instruct-AWQ',"
        f"  local_dir='{MODEL_DIR}',"
        f"  ignore_patterns=['*.msgpack','*.h5','flax_model*','*.ot'],"
        f")\""
    )
    .run_commands(
        "python -c \"import easyocr; easyocr.Reader(['en'], gpu=False, verbose=False)\""
    )
    .add_local_file("inference.py", f"{WORKDIR}/inference.py")
    .add_local_dir(r"c:\Users\PRABHAKAR\Documents\GNR_Final_project\test_dir\patches", f"{TEST_DIR}/patches")
    .add_local_file(r"c:\Users\PRABHAKAR\Documents\GNR_Final_project\test_dir\test.csv", f"{TEST_DIR}/test.csv")
)


@app.function(image=image, gpu="A100-80GB", timeout=3600)
def run_inference_bot():
    import subprocess, os

    os.chdir(WORKDIR)

    print("\n" + "=" * 80)
    print("GRADING BOT TEST — OFFLINE INFERENCE")
    print("=" * 80)

    model_dir = Path(MODEL_DIR)
    if model_dir.exists():
        n = len(list(model_dir.iterdir()))
        print(f"Model weights: {n} files in {model_dir}")
    else:
        print(f"Model weights NOT found: {model_dir}")
        return {"status": "FAILED", "error": "Model not found"}

    test_dir = Path(TEST_DIR)
    patches = test_dir / "patches"
    test_csv = test_dir / "test.csv"
    if patches.exists() and test_csv.exists():
        n_patches = len(list(patches.glob("*.png")))
        print(f"Test data: {n_patches} patches, test.csv present")
    else:
        print(f"Test data NOT found at {test_dir}")
        return {"status": "FAILED", "error": "Test data not found"}

    print(f"\nRunning: python inference.py --test_dir {TEST_DIR}")
    print("-" * 80)

    result = subprocess.run(
        ["python", "inference.py", "--test_dir", TEST_DIR],
        capture_output=True, text=True, timeout=1800,
    )

    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr[-1000:])
    print(f"Exit code: {result.returncode}")

    submission = Path("submission.csv")
    if result.returncode == 0 and submission.exists():
        import csv

        # Read submission
        preds = {}
        with open("submission.csv") as f:
            for row in csv.DictReader(f):
                preds[row["id"]] = int(row["option"])

        # Read ground truth from test.csv (has correct_answer column)
        truth = {}
        test_csv_path = f"{TEST_DIR}/test.csv"
        with open(test_csv_path) as f:
            reader = csv.DictReader(f)
            if "correct_answer" in reader.fieldnames:
                for row in reader:
                    truth[row["id"]] = int(row["correct_answer"])

        # Score
        correct = 0
        wrong = 0
        unanswered = 0
        hallucinated = 0
        details = []

        for qid in sorted(preds.keys(), key=lambda x: int(x.split("_")[1])):
            pred = preds[qid]
            gt = truth.get(qid)

            if gt is None:
                details.append(f"  {qid}: pred={pred} (no ground truth)")
                continue

            if pred == 5:
                unanswered += 1
                details.append(f"  {qid}: pred=5 (SKIP)  correct={gt}")
            elif pred not in (1, 2, 3, 4):
                hallucinated += 1
                details.append(f"  {qid}: pred={pred} (HALLUCINATED)  correct={gt}")
            elif pred == gt:
                correct += 1
                details.append(f"  {qid}: pred={pred} == {gt} CORRECT")
            else:
                wrong += 1
                details.append(f"  {qid}: pred={pred} != {gt} WRONG")

        total = len(preds)
        score = correct - 0.25 * wrong - hallucinated

        print("\n" + "=" * 80)
        print("EVALUATION RESULTS")
        print("=" * 80)
        print(f"\n  Total questions : {total}")
        print(f"  Correct         : {correct}")
        print(f"  Wrong           : {wrong}")
        print(f"  Unanswered (5)  : {unanswered}")
        print(f"  Hallucinated    : {hallucinated}")
        print(f"\n  Score = {correct} - 0.25*{wrong} - {hallucinated} = {score:.2f} / {total}")
        print(f"  Accuracy        : {correct}/{total} = {100*correct/total:.1f}%")
        print("\n" + "-" * 80)
        print("QUESTION-BY-QUESTION:")
        for d in details:
            print(d)

        return {
            "status": "SUCCESS",
            "rows": total,
            "correct": correct,
            "wrong": wrong,
            "unanswered": unanswered,
            "hallucinated": hallucinated,
            "score": score,
            "accuracy_pct": round(100 * correct / total, 1) if total else 0,
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

    print("\nModal bot test — Qwen2.5-VL-72B on A100")
    print("-" * 60)

    result = run_inference_bot.remote()

    print(json.dumps(result, indent=2))
    if result["status"] == "SUCCESS":
        print(f"\nBOT TEST PASSED")
        print(f"  Score: {result.get('score', '?')} / {result.get('rows', '?')}")
        print(f"  Correct: {result.get('correct', '?')}  Wrong: {result.get('wrong', '?')}  Skipped: {result.get('unanswered', '?')}")
    else:
        print("\nBOT TEST FAILED")
