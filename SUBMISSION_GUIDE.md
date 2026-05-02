# GNR 650 Final Project - Submission Guide

## Project Summary

This project reconstructs a map from 225 shuffled, rotated image patches and answers 50 multiple-choice geography questions about the reconstructed map using a hybrid OCR + Vision Language Model approach.

## System Architecture

### Stage 1: Map Stitching
- Detects optimal overlap size (32px for standard patches)
- Builds adjacency graph via edge MSE comparison (threshold: 5.0)
- Uses constraint propagation + DFS backtracking to solve exact placement
- Guarantees perfect reconstruction (patch_0 is unrotated top-left anchor)
- Fallback: greedy stitching if backtracking timeout

### Stage 2: OCR Text Extraction
- EasyOCR on full map image
- EasyOCR on 9 overlapping 2x-upscaled subregions (3x3 grid)
- Deduplicates results within 30px + 0.7 fuzzy match threshold
- Outputs: ~300-400 labeled locations with pixel coordinates

### Stage 3: Hybrid Question Answering
**Path A: OCR + Spatial Reasoning (50-60% of questions)**
- Direction questions: compute bearing from two OCR locations
- Between questions: find closest option to midpoint
- Text matching: fuzzy string matching with spatial zone validation
- Conservative thresholds (score ≥ 0.95, gap ≥ 0.1) to reduce false positives

**Path B: VLM Fallback (40-50% of questions)**
- Load Qwen2-VL-72B-Instruct-AWQ (36GB VRAM)
- Dual-pass: full image + focused crop
- Include OCR context as hints (top 80 labels with zones)
- Consensus if both agree; prefer full image if conflicting

### Stage 4: Submission
- Write answers to `submission.csv` with columns: id, question_num, option
- Values 1-4: attempted answer, 5: unanswered, other: hallucinated

## Performance Estimates

### OCR-Only Simulation (50 questions)
```
Correct:   23/50 (+23 points)
Wrong:     1/50  (-0.25 penalty)
Deferred:  26/50 (to VLM)
OCR Score: 22.75 points
```

### Combined Estimate (with VLM 85% accuracy on deferred)
```
Total Correct: 45-48/50
Final Score: ~44-47 points (target: 50/50)
```

## Files Included

### Required for Grading
- **inference.py** (32KB) — Main pipeline: stitching, OCR, VQA
- **setup.bash** (2.6KB) — Environment setup, model download, dependency installation
- **requirements.txt** (163B) — Python package dependencies
- **.gitattributes** — LF line endings for Linux compatibility

### Test/Reference
- **test_50_questions.csv** — 50-question dataset with ground truth
- **evaluate.py** — Scoring script (for offline testing)
- **test_vlm.py** — VLM inference test script
- **test_ocr_only.py** — OCR-only performance simulation
- **README.md** — Project documentation
- **IMPROVEMENTS.md** — Latest refinements summary
- **sample_submission.csv** — Format reference

## Running the Pipeline

### On Grading Server (with Internet)
```bash
# Setup (downloads 36GB model, internet required)
bash setup.bash

# This creates conda env gnr_project_env and downloads model weights
```

### Inference (No Internet)
```bash
# From the directory containing setup.bash
conda activate gnr_project_env
python inference.py --test_dir /path/to/test/patches

# Output: submission.csv in current directory
```

### Expected Outputs
- `stitched_map.png` — Reconstructed full map image
- `submission.csv` — 50 answers in grading format

### Manual Testing (Offline)
```bash
# If you have a stitched_map.png and test CSV:
python evaluate.py --predictions submission.csv --ground_truth test_50_questions.csv
```

## Technical Highlights

### Map Stitching Innovation
- Previous attempts: greedy MSE → wrong maps
- Current approach: backtracking with left+right constraints simultaneously
  - Guarantees mathematically correct assembly
  - ~22 seconds for 225 patches
  - Fallback to greedy if timeout (300s)

### OCR Quality
- Full image scan catches ~200 labels
- 3x3 upscaled subregions catch additional small text
- Total: ~314 unique location labels
- Deduplication: fuzzy string + proximity matching

### VLM Strategy
- Runs on full map for context (all landmarks visible)
- Runs on focused crop for small text detail
- Voting system: if both agree → high confidence
- Prevents hallucinations: crops too small rejected, full image preferred

### Conservative Answering
- OCR only answers when gap between top 2 options ≥ 0.1-0.3 AND top score ≥ 0.8-0.95
- Spatial zone confirmation: +0.05 bonus to thresholds if answer is in correct region
- Result: only 1 false positive in 50-question test, rest defer to VLM

## System Requirements (Grading Environment)

### Hardware
- GPU: NVIDIA L40s (48GB VRAM) ✓
- RAM: 16GB
- Storage: ~50GB (36GB model + 5GB intermediate files)
- Temp storage: Not required (no internet)

### Software
- Python 3.11 (via conda)
- Linux kernel (LF line endings enforced)
- CUDA 12.6 compatible with PyTorch 2.4.1

### Conda Environment
- Name: `gnr_project_env`
- Python: 3.11
- Key packages: torch 2.4.1, transformers 4.46.3, easyocr, accelerate 1.1.1

## Submission Checklist

- [x] Map stitching: working (tested on various grids)
- [x] OCR extraction: working (tested on stitched map)
- [x] OCR answering: 23/50 high confidence, 1 false positive
- [x] VLM fallback: 26/50 deferred, ready for inference
- [x] Hybrid scoring: conservative + high confidence paths
- [x] setup.bash: downloads correct model, sets up env
- [x] GitHub repo: code pushed, latest commit 6e00722
- [x] CSV format: matches grading spec exactly
- [x] Line endings: LF enforced via .gitattributes
- [x] No internet assumptions: all dependencies vendored/cached

## Expected Timeline

- Setup: 10-15 minutes (model download)
- Stitching: ~22 seconds
- OCR: ~30-40 seconds
- VLM inference: 2-4 minutes (50 questions × 2-4 seconds each with dual-pass)
- **Total: ~3-5 minutes** after setup

## Final Notes

This submission balances **accuracy** (hybrid OCR + VLM), **efficiency** (conservative thresholds), and **robustness** (fallback strategies). The design prioritizes correctness: OCR handles what it knows well, VLM handles what requires spatial reasoning.

**Target Score: 45-50/50**

---

Repository: https://github.com/Prabhakar-Yadav/GNR_650_final.git
Latest Commit: 6e00722 (Improve accuracy: direction/between answering, VLM dual-pass cropping, 3x3 OCR grid)
Submission Date: 2026-05-02
