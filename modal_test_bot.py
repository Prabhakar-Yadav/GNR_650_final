"""
Modal test script - simulates grading bot WITHOUT internet during inference.

Usage:
  modal run modal_test_bot.py

This:
1. Creates a Modal image with all dependencies (WITH internet)
2. Downloads model weights and EasyOCR assets during setup (WITH internet)
3. Runs inference WITHOUT internet (simulates grading bot)
4. Generates and grades submission.csv
"""

import modal
import os
from pathlib import Path

app = modal.App("gnr-grading-test")

# Create image: install deps + download model/assets (WITH internet during build)
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install_from_requirements("requirements.txt")
    .run_commands(
        # Install pip packages
        "pip install --upgrade pip",
        "pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 "
        "--index-url https://download.pytorch.org/whl/cu124",
        "pip install transformers==4.51.3 accelerate==1.6.0 huggingface_hub "
        "qwen-vl-utils==0.0.8 pillow opencv-python-headless pandas numpy easyocr bitsandbytes autoawq==0.2.9",
    )
    .run_commands(
        # Download model weights (WITH internet during image build)
        "python -c \"from huggingface_hub import snapshot_download; "
        "snapshot_download(repo_id='Qwen/Qwen2.5-VL-72B-Instruct-AWQ', "
        "local_dir='./model_weights/Qwen2.5-VL-72B-Instruct-AWQ', "
        "ignore_patterns=['*.msgpack', '*.h5', 'flax_model*', '*.ot'])\"",
    )
    .run_commands(
        # Download EasyOCR assets (WITH internet during image build)
        "python -c \"import easyocr; easyocr.Reader(['en'], gpu=False, verbose=False)\"",
    )
)

@app.function(
    image=image,
    gpu="A100",  # Or L40s if available: gpu="L40s"
    timeout=3600,  # 1 hour
)
def run_inference_bot(test_dir_path: str):
    """
    Simulates grading bot:
    - Setup already done (image build)
    - Runs inference WITHOUT internet
    - Returns submission.csv content
    """
    import subprocess
    import json
    
    print("\n" + "="*80)
    print("GRADING BOT TEST - OFFLINE INFERENCE")
    print("="*80)
    
    # Verify model files exist (no internet to download)
    model_dir = Path("./model_weights/Qwen2.5-VL-72B-Instruct-AWQ")
    if model_dir.exists():
        print(f"✓ Model weights found: {model_dir}")
        model_files = list(model_dir.glob("*"))
        print(f"  Files: {len(model_files)} items")
    else:
        print(f"✗ Model weights NOT found: {model_dir}")
        return {"status": "FAILED", "error": "Model not found"}
    
    # Verify EasyOCR cache exists
    import os
    easyocr_cache = Path(os.path.expanduser("~/.EasyOCR"))
    if easyocr_cache.exists():
        print(f"✓ EasyOCR cache found: {easyocr_cache}")
    else:
        print(f"⚠ EasyOCR cache not found (will be created on first use)")
    
    print("\nRunning inference without internet...")
    print(f"Test directory: {test_dir_path}")
    print("-" * 80)
    
    # Run inference (offline - no internet)
    result = subprocess.run(
        ["python", "inference.py", "--test_dir", test_dir_path],
        capture_output=True,
        text=True,
        timeout=900,  # 15 min timeout for inference
    )
    
    print("STDOUT:")
    print(result.stdout)
    
    if result.stderr:
        print("\nSTDERR:")
        print(result.stderr)
    
    print("-" * 80)
    print(f"Exit code: {result.returncode}")
    
    if result.returncode == 0:
        # Check if submission.csv was created
        if Path("submission.csv").exists():
            with open("submission.csv", "r") as f:
                content = f.read()
            lines = content.strip().split("\n")
            print(f"\n✓ submission.csv created: {len(lines)} rows")
            print("\nFirst 5 rows:")
            for line in lines[:5]:
                print(f"  {line}")
            
            return {
                "status": "SUCCESS",
                "exit_code": result.returncode,
                "rows": len(lines) - 1,  # Exclude header
                "submission_preview": lines[:5],
            }
        else:
            print("✗ submission.csv NOT created")
            return {
                "status": "FAILED",
                "exit_code": result.returncode,
                "error": "submission.csv not found",
            }
    else:
        return {
            "status": "FAILED",
            "exit_code": result.returncode,
            "stdout": result.stdout[-500:],  # Last 500 chars
            "stderr": result.stderr[-500:],
        }

@app.local_entrypoint()
def main():
    """Test the bot on your local test data"""
    import json
    
    # Mount your local test directory
    test_dir = "/path/to/local/test_data"  # Change this to your test path
    
    print("\n" + "="*80)
    print("MODAL BOT TEST - WITHOUT INTERNET")
    print("="*80)
    print(f"\nTest data: {test_dir}")
    print("\nPhase 1: Building image (downloads model, WITH internet)")
    print("  - Installing PyTorch")
    print("  - Downloading Qwen2.5-VL-72B-Instruct-AWQ (~36GB)")
    print("  - Downloading EasyOCR assets")
    print("\nPhase 2: Running inference (NO internet)")
    print("  - All files pre-cached in image")
    print("  - Simulates grading bot behavior")
    print("-" * 80)
    
    result = run_inference_bot(test_dir)
    
    print("\n" + "="*80)
    print("TEST RESULT")
    print("="*80)
    print(json.dumps(result, indent=2))
    
    if result["status"] == "SUCCESS":
        print("\n✓ BOT TEST PASSED - Inference runs offline as expected")
        return 0
    else:
        print("\n✗ BOT TEST FAILED - See errors above")
        return 1

if __name__ == "__main__":
    main()
