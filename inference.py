"""
Map reconstruction + VQA inference script.
Usage: python inference.py --test_dir <path_to_test_dir>
Outputs: submission.csv in current working directory
"""

import argparse
import os
import csv
import math
import numpy as np
import cv2
import torch
import pandas as pd
from pathlib import Path
from PIL import Image


def load_patches(patches_dir):
    patches = {}
    for fname in os.listdir(patches_dir):
        if fname.endswith(".png"):
            idx = int(fname.replace("patch_", "").replace(".png", ""))
            img = cv2.imread(os.path.join(patches_dir, fname))
            patches[idx] = img
    return patches


def rotate_image(img, k):
    if k == 0:
        return img
    elif k == 1:
        return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    elif k == 2:
        return cv2.rotate(img, cv2.ROTATE_180)
    elif k == 3:
        return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)


def detect_overlap(patches):
    """Auto-detect the overlap between adjacent patches by testing powers of 2."""
    p0 = patches[0]
    h, w = p0.shape[:2]
    for ov in [64, 48, 32, 24, 16, 8, 4, 2, 1]:
        if ov >= w:
            continue
        right_strip = p0[:, -ov:, :].astype(np.float32)
        for idx in range(1, len(patches)):
            img = patches[idx]
            for k in range(4):
                rot = rotate_image(img, k)
                left_strip = rot[:, :ov, :].astype(np.float32)
                mse = np.mean((right_strip - left_strip) ** 2)
                if mse < 1.0:
                    return ov
    return 0


def stitch_patches(patches, n_rows, n_cols, overlap):
    """
    Stitch patches using overlap-based constraint propagation with backtracking.
    patch_0 is always top-left (no rotation).
    """
    THRESH = 5.0
    n_patches = len(patches)

    # Precompute edge strips for all patch/rotation combos
    edges = {}
    for idx in range(n_patches):
        img = patches[idx]
        for k in range(4):
            rot = rotate_image(img, k)
            edges[(idx, k)] = (
                rot[:, :overlap, :].astype(np.float32),   # left
                rot[:, -overlap:, :].astype(np.float32),  # right
                rot[:overlap, :, :].astype(np.float32),   # top
                rot[-overlap:, :, :].astype(np.float32),  # bottom
            )

    # Build adjacency graph
    right_adj = {}
    bottom_adj = {}
    for idx1 in range(n_patches):
        for k1 in range(4):
            key1 = (idx1, k1)
            _, right1, _, bottom1 = edges[key1]
            r_matches = []
            b_matches = []
            for idx2 in range(n_patches):
                if idx2 == idx1:
                    continue
                for k2 in range(4):
                    key2 = (idx2, k2)
                    left2, _, top2, _ = edges[key2]
                    if np.mean((right1 - left2) ** 2) < THRESH:
                        r_matches.append(key2)
                    if np.mean((bottom1 - top2) ** 2) < THRESH:
                        b_matches.append(key2)
            right_adj[key1] = r_matches
            bottom_adj[key1] = b_matches

    # Solve placement with backtracking
    grid = [[None] * n_cols for _ in range(n_rows)]
    grid[0][0] = (0, 0)

    def get_candidates(row, col, used_patches):
        candidates = None
        if col > 0 and grid[row][col - 1] is not None:
            left_key = grid[row][col - 1]
            r_set = set((i, k) for (i, k) in right_adj[left_key] if i not in used_patches)
            candidates = r_set if candidates is None else candidates & r_set
        if row > 0 and grid[row - 1][col] is not None:
            top_key = grid[row - 1][col]
            b_set = set((i, k) for (i, k) in bottom_adj[top_key] if i not in used_patches)
            candidates = b_set if candidates is None else candidates & b_set
        return candidates if candidates is not None else set()

    def solve(pos, used_patches):
        if pos == n_patches:
            return True
        row = pos // n_cols
        col = pos % n_cols
        if row == 0 and col == 0:
            return solve(pos + 1, used_patches)
        candidates = get_candidates(row, col, used_patches)
        for cand in candidates:
            idx, k = cand
            grid[row][col] = cand
            used_patches.add(idx)
            if solve(pos + 1, used_patches):
                return True
            used_patches.remove(idx)
            grid[row][col] = None
        return False

    used = {0}
    success = solve(0, used)

    if not success:
        print("WARNING: Backtracking failed, falling back to greedy.")
        return stitch_patches_greedy(patches, n_rows, n_cols, overlap)

    # Assemble final image
    stride = patches[0].shape[0] - overlap
    height = stride * (n_rows - 1) + patches[0].shape[0]
    width = stride * (n_cols - 1) + patches[0].shape[1]
    result = np.zeros((height, width, 3), dtype=np.uint8)
    for r in range(n_rows):
        for c in range(n_cols):
            idx, k = grid[r][c]
            img = rotate_image(patches[idx], k)
            y = r * stride
            x = c * stride
            result[y:y + img.shape[0], x:x + img.shape[1]] = img
    return result


def stitch_patches_greedy(patches, n_rows, n_cols, overlap):
    """Fallback greedy stitcher if backtracking fails."""
    n_patches = len(patches)
    grid = [[None] * n_cols for _ in range(n_rows)]
    used = set()
    grid[0][0] = patches[0]
    used.add(0)

    def edge_mse(a, b, side):
        if side == "right":
            ea = a[:, -overlap:, :].astype(np.float32)
            eb = b[:, :overlap, :].astype(np.float32)
        elif side == "bottom":
            ea = a[-overlap:, :, :].astype(np.float32)
            eb = b[:overlap, :, :].astype(np.float32)
        return np.mean((ea - eb) ** 2)

    for row in range(n_rows):
        for col in range(n_cols):
            if row == 0 and col == 0:
                continue
            best_idx, best_rot, best_score = None, 0, float("inf")
            for idx, img in patches.items():
                if idx in used:
                    continue
                for k in range(4):
                    rotated = rotate_image(img, k)
                    score = 0.0
                    count = 0
                    if col > 0 and grid[row][col - 1] is not None:
                        score += edge_mse(grid[row][col - 1], rotated, "right")
                        count += 1
                    if row > 0 and grid[row - 1][col] is not None:
                        score += edge_mse(grid[row - 1][col], rotated, "bottom")
                        count += 1
                    if count > 0:
                        score /= count
                    if score < best_score:
                        best_score = score
                        best_idx = idx
                        best_rot = k
            if best_idx is not None:
                grid[row][col] = rotate_image(patches[best_idx], best_rot)
                used.add(best_idx)

    stride = patches[0].shape[0] - overlap
    height = stride * (n_rows - 1) + patches[0].shape[0]
    width = stride * (n_cols - 1) + patches[0].shape[1]
    result = np.zeros((height, width, 3), dtype=np.uint8)
    for r in range(n_rows):
        for c in range(n_cols):
            if grid[r][c] is not None:
                y = r * stride
                x = c * stride
                h, w = grid[r][c].shape[:2]
                result[y:y + h, x:x + w] = grid[r][c]
    return result


def answer_questions_with_vlm(map_image_path, test_csv_path, model_name=None):
    """Use Qwen2-VL to answer multiple-choice questions about the reconstructed map."""
    from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
    from qwen_vl_utils import process_vision_info

    local_weights = Path("./model_weights/Qwen2-VL-7B-Instruct")
    if model_name is None:
        if local_weights.exists():
            model_name = str(local_weights)
        else:
            raise FileNotFoundError(
                f"Model weights not found at {local_weights}. "
                "Run setup.bash first to download them."
            )

    print(f"Loading model: {model_name}")
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(model_name)

    df = pd.read_csv(test_csv_path)
    results = []

    pil_image = Image.open(map_image_path).convert("RGB")

    for _, row in df.iterrows():
        qid = row["id"]
        question = row["question"]
        opt1 = row["option_1"]
        opt2 = row["option_2"]
        opt3 = row["option_3"]
        opt4 = row["option_4"]

        prompt = (
            f"You are analyzing a satellite/map image. Answer the following multiple choice question.\n\n"
            f"Question: {question}\n\n"
            f"Options:\n"
            f"1. {opt1}\n"
            f"2. {opt2}\n"
            f"3. {opt3}\n"
            f"4. {opt4}\n\n"
            f"Look carefully at the map image and respond with ONLY the number (1, 2, 3, or 4) of the correct answer. "
            f"If you are not sure, respond with 5."
        )

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": pil_image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(model.device)

        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=16, temperature=0.0, do_sample=False)

        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0].strip()

        answer = 5
        for ch in output_text:
            if ch in "12345":
                answer = int(ch)
                break

        print(f"  {qid}: {output_text!r} -> {answer}")
        results.append({"id": qid, "question_num": qid, "option": answer})

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_dir", required=True, help="Path to test directory")
    args = parser.parse_args()

    test_dir = Path(args.test_dir)
    patches_dir = test_dir / "patches"
    test_csv = test_dir / "test.csv"

    print("Loading patches...")
    patches = load_patches(str(patches_dir))
    n_patches = len(patches)

    # Determine grid dimensions
    grid_size = int(math.isqrt(n_patches))
    if grid_size * grid_size == n_patches:
        n_rows, n_cols = grid_size, grid_size
    else:
        best = (1, n_patches)
        for r in range(1, int(n_patches ** 0.5) + 1):
            if n_patches % r == 0:
                c = n_patches // r
                if abs(r - c) < abs(best[0] - best[1]):
                    best = (r, c)
        n_rows, n_cols = best
    print(f"  {n_patches} patches -> {n_rows}x{n_cols} grid")

    print("Detecting overlap...")
    overlap = detect_overlap(patches)
    print(f"  Overlap: {overlap}px")

    print("Stitching map...")
    full_map = stitch_patches(patches, n_rows, n_cols, overlap)
    map_path = "stitched_map.png"
    cv2.imwrite(map_path, full_map)
    print(f"  Saved stitched map: {map_path} ({full_map.shape})")

    print("Answering questions with VLM...")
    results = answer_questions_with_vlm(map_path, str(test_csv))

    print("Writing submission.csv...")
    with open("submission.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "question_num", "option"])
        writer.writeheader()
        writer.writerows(results)

    print(f"Done. submission.csv written with {len(results)} rows.")


if __name__ == "__main__":
    main()
