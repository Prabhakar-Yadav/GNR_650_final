"""
Map reconstruction + VQA inference script.
Usage: python inference.py --test_dir <path_to_test_dir>
Outputs: submission.csv in current working directory
"""

import argparse
import os
import sys
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
    """Rotate image by k*90 degrees clockwise."""
    if k == 0:
        return img
    elif k == 1:
        return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    elif k == 2:
        return cv2.rotate(img, cv2.ROTATE_180)
    elif k == 3:
        return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)


def edge_mse(a, b, side):
    """
    Compute MSE between edge of a and matching edge of b.
    side: 'right' means right edge of a vs left edge of b
          'bottom' means bottom edge of a vs top edge of b
    """
    if side == "right":
        ea = a[:, -1, :].astype(np.float32)
        eb = b[:, 0, :].astype(np.float32)
    elif side == "bottom":
        ea = a[-1, :, :].astype(np.float32)
        eb = b[0, :, :].astype(np.float32)
    return np.mean((ea - eb) ** 2)


def stitch_patches(patches, n_rows, n_cols):
    """
    Greedily stitch patches into an n_rows x n_cols grid.
    patch_0 is always top-left (no rotation).
    Returns the stitched image.
    """
    grid = [[None] * n_cols for _ in range(n_rows)]
    used = set()

    # Place patch_0 at top-left without rotation
    grid[0][0] = patches[0]
    used.add(0)

    # Fill positions row by row
    for row in range(n_rows):
        for col in range(n_cols):
            if row == 0 and col == 0:
                continue

            best_idx = None
            best_rot = 0
            best_score = float("inf")

            for idx, img in patches.items():
                if idx in used:
                    continue
                for k in range(4):
                    rotated = rotate_image(img, k)
                    score = 0.0
                    count = 0
                    # Check left neighbor
                    if col > 0 and grid[row][col - 1] is not None:
                        score += edge_mse(grid[row][col - 1], rotated, "right")
                        count += 1
                    # Check top neighbor
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

    # Assemble full image
    rows_imgs = []
    for row in range(n_rows):
        row_imgs = [grid[row][col] for col in range(n_cols)]
        rows_imgs.append(np.concatenate(row_imgs, axis=1))
    full_map = np.concatenate(rows_imgs, axis=0)
    return full_map


def answer_questions_with_vlm(map_image_path, test_csv_path, model_name=None):
    """Use a VLM to answer multiple-choice questions about the map."""
    from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
    from qwen_vl_utils import process_vision_info

    # Weights are downloaded to ./model_weights/ by setup.bash (same dir inference.py runs from).
    # No internet is available at inference time, so we must use the local path.
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

        # Extract the digit answer
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

    # Determine grid dimensions — prefer square, fall back to best rectangle
    grid_size = int(math.isqrt(n_patches))
    if grid_size * grid_size == n_patches:
        n_rows, n_cols = grid_size, grid_size
    else:
        # Find the factor pair closest to square
        best = (1, n_patches)
        for r in range(1, int(n_patches**0.5) + 1):
            if n_patches % r == 0:
                c = n_patches // r
                if abs(r - c) < abs(best[0] - best[1]):
                    best = (r, c)
        n_rows, n_cols = best
    print(f"  {n_patches} patches -> {n_rows}x{n_cols} grid")

    print("Stitching map...")
    full_map = stitch_patches(patches, n_rows, n_cols)
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
