"""
Map reconstruction + VQA inference script.
Hybrid approach: OCR text extraction + spatial reasoning + VLM fallback.
Usage: python inference.py --test_dir <path_to_test_dir>
Outputs: submission.csv in current working directory
"""

import argparse
import os
import csv
import math
import re
import numpy as np
import cv2
import torch
import pandas as pd
from pathlib import Path
from PIL import Image
from difflib import SequenceMatcher


# ═══════════════════════════════════════════════════════════════════════════════
# MAP STITCHING
# ═══════════════════════════════════════════════════════════════════════════════

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
    p0 = patches[0]
    h, w = p0.shape[:2]
    for ov in [64, 48, 32, 24, 16, 8, 4, 2]:
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
    return 1


def stitch_patches(patches, n_rows, n_cols, overlap):
    import time as _time
    THRESH = 5.0
    TIMEOUT = 300
    n_patches = len(patches)

    edges = {}
    for idx in range(n_patches):
        img = patches[idx]
        for k in range(4):
            rot = rotate_image(img, k)
            edges[(idx, k)] = (
                rot[:, :overlap, :].astype(np.float32),
                rot[:, -overlap:, :].astype(np.float32),
                rot[:overlap, :, :].astype(np.float32),
                rot[-overlap:, :, :].astype(np.float32),
            )

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

    grid = [[None] * n_cols for _ in range(n_rows)]
    grid[0][0] = (0, 0)
    _start_time = _time.time()

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
        if _time.time() - _start_time > TIMEOUT:
            return False
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


# ═══════════════════════════════════════════════════════════════════════════════
# OCR-BASED TEXT EXTRACTION
# ═══════════════════════════════════════════════════════════════════════════════

def extract_text_from_map(map_image_path):
    """Extract all text labels + their pixel coordinates using EasyOCR."""
    import easyocr

    reader = easyocr.Reader(["en"], gpu=torch.cuda.is_available(), verbose=False)
    img = cv2.imread(map_image_path)
    h, w = img.shape[:2]

    all_texts = []

    def run_ocr_on_image(image, offset_x=0, offset_y=0, scale=1.0):
        results = reader.readtext(image)
        for (bbox, text, conf) in results:
            cx = (sum(p[0] for p in bbox) / 4) / scale + offset_x
            cy = (sum(p[1] for p in bbox) / 4) / scale + offset_y
            all_texts.append({
                "text": text,
                "confidence": conf,
                "cx": cx, "cy": cy,
                "norm_x": cx / w, "norm_y": cy / h,
            })

    # Run OCR on full image
    run_ocr_on_image(img)

    # Also run OCR on upscaled quadrants for small text
    quadrants = [
        (0, 0, w // 2, h // 2),
        (w // 2, 0, w, h // 2),
        (0, h // 2, w // 2, h),
        (w // 2, h // 2, w, h),
    ]
    for x1, y1, x2, y2 in quadrants:
        crop = img[y1:y2, x1:x2]
        upscaled = cv2.resize(crop, (crop.shape[1] * 2, crop.shape[0] * 2),
                              interpolation=cv2.INTER_CUBIC)
        run_ocr_on_image(upscaled, offset_x=x1, offset_y=y1, scale=2.0)

    # Deduplicate
    deduped = []
    for t in all_texts:
        is_dup = False
        for d in deduped:
            if (abs(t["cx"] - d["cx"]) < 30 and abs(t["cy"] - d["cy"]) < 30
                    and fuzzy_ratio(t["text"].lower(), d["text"].lower()) > 0.7):
                if t["confidence"] > d["confidence"]:
                    d.update(t)
                is_dup = True
                break
        if not is_dup:
            deduped.append(t)

    return deduped, h, w


def fuzzy_ratio(a, b):
    return SequenceMatcher(None, a, b).ratio()


def find_text_on_map(ocr_texts, query, threshold=0.5):
    """Find OCR text entries that match a query string. Returns list of (text_entry, score)."""
    query_lower = query.lower().strip()
    query_words = set(w for w in query_lower.split() if len(w) > 2)
    matches = []
    for t in ocr_texts:
        text_lower = t["text"].lower().strip()
        # Skip very short OCR results (noise)
        if len(text_lower) < 3:
            continue
        # Exact full match
        if query_lower == text_lower:
            score = 1.0
        # Substantial substring match (both must be reasonably long)
        elif len(query_lower) >= 4 and len(text_lower) >= 4:
            if query_lower in text_lower:
                score = len(query_lower) / len(text_lower)
                score = max(score, 0.8)
            elif text_lower in query_lower:
                score = len(text_lower) / len(query_lower)
                score = max(score, 0.7)
            else:
                score = fuzzy_ratio(query_lower, text_lower)
        else:
            score = fuzzy_ratio(query_lower, text_lower)
        # Word overlap bonus
        text_words = set(w for w in text_lower.split() if len(w) > 2)
        if query_words and text_words:
            word_overlap = len(query_words & text_words) / len(query_words)
            if word_overlap > 0.3:
                score = max(score, 0.5 + word_overlap * 0.5)
        if score >= threshold:
            matches.append((t, score))
    matches.sort(key=lambda x: -x[1])
    return matches


# ═══════════════════════════════════════════════════════════════════════════════
# OCR-BASED QUESTION ANSWERING
# ═══════════════════════════════════════════════════════════════════════════════

def get_spatial_zone(norm_x, norm_y):
    """Convert normalized coordinates to spatial description."""
    zones = []
    if norm_y < 0.33:
        zones.append("north")
    elif norm_y > 0.66:
        zones.append("south")
    if norm_x < 0.33:
        zones.append("west")
    elif norm_x > 0.66:
        zones.append("east")
    if not zones:
        zones.append("center")
    return zones


def answer_with_ocr(question, options, ocr_texts, map_h, map_w):
    """
    Try to answer a question using OCR text + spatial reasoning.
    Returns (answer_idx_1based, confidence) or (None, 0) if can't answer.
    """
    q_lower = question.lower()

    # Strategy 1: Which option text appears on the map?
    option_scores = []
    for i, opt in enumerate(options):
        matches = find_text_on_map(ocr_texts, opt, threshold=0.45)
        best_score = matches[0][1] if matches else 0
        best_match = matches[0][0] if matches else None
        option_scores.append((i + 1, best_score, best_match, opt))

    # Sort by match score descending
    option_scores.sort(key=lambda x: -x[1])

    # If one option clearly matches and others don't, use it
    if option_scores[0][1] >= 0.55:
        top_score = option_scores[0][1]
        second_score = option_scores[1][1] if len(option_scores) > 1 else 0

        # Spatial constraint check
        spatial_keywords = extract_spatial_keywords(q_lower)
        if spatial_keywords and option_scores[0][2]:
            match_entry = option_scores[0][2]
            zones = get_spatial_zone(match_entry["norm_x"], match_entry["norm_y"])
            spatial_ok = check_spatial_match(spatial_keywords, zones)
            if not spatial_ok:
                # Top match doesn't satisfy spatial constraint
                # Check if any other option does
                for idx, score, match, opt in option_scores[1:]:
                    if score >= 0.45 and match:
                        zones2 = get_spatial_zone(match["norm_x"], match["norm_y"])
                        if check_spatial_match(spatial_keywords, zones2):
                            return idx, 0.7
                # Still return top match but lower confidence
                return option_scores[0][0], 0.5

        # Clear winner
        if top_score - second_score > 0.15:
            return option_scores[0][0], min(0.95, top_score)

        # Multiple matches - use spatial to disambiguate
        if spatial_keywords:
            for idx, score, match, opt in option_scores:
                if score >= 0.45 and match:
                    zones = get_spatial_zone(match["norm_x"], match["norm_y"])
                    if check_spatial_match(spatial_keywords, zones):
                        return idx, 0.75

        # Just return best match
        if top_score >= 0.55:
            return option_scores[0][0], top_score * 0.8

    # Strategy 2: "near" / "between" / proximity questions
    if any(kw in q_lower for kw in ["near", "close", "adjacent", "nearest", "next to"]):
        subject = extract_subject_near(q_lower)
        if subject:
            subject_matches = find_text_on_map(ocr_texts, subject, threshold=0.35)
            if subject_matches:
                subject_loc = subject_matches[0][0]
                min_dist = float("inf")
                best_opt = None
                found_any = False
                for i, opt in enumerate(options):
                    opt_matches = find_text_on_map(ocr_texts, opt, threshold=0.40)
                    if opt_matches:
                        found_any = True
                        opt_loc = opt_matches[0][0]
                        dist = ((subject_loc["cx"] - opt_loc["cx"]) ** 2 +
                                (subject_loc["cy"] - opt_loc["cy"]) ** 2) ** 0.5
                        if dist < min_dist:
                            min_dist = dist
                            best_opt = i + 1
                if best_opt is not None and found_any:
                    return best_opt, 0.8

    # Strategy 3: Directional questions - "direction of X from Y"
    if any(d in q_lower for d in ["direction", "north-east", "south-west", "north", "south", "east", "west"]):
        # Check if it's asking about relative direction
        for i, opt in enumerate(options):
            opt_lower = opt.lower()
            if "north" in opt_lower or "south" in opt_lower or "east" in opt_lower or "west" in opt_lower:
                # This is likely a direction answer
                # Find the two locations mentioned and compute direction
                pass  # Handled by spatial zone matching above

    return None, 0


def extract_spatial_keywords(question):
    keywords = []
    spatial_terms = ["north", "south", "east", "west", "center", "central",
                     "top", "bottom", "left", "right", "upper", "lower"]
    for term in spatial_terms:
        if term in question:
            keywords.append(term)
    return keywords


def check_spatial_match(keywords, zones):
    mapping = {
        "north": "north", "upper": "north", "top": "north",
        "south": "south", "lower": "south", "bottom": "south",
        "east": "east", "right": "east", "eastern": "east",
        "west": "west", "left": "west", "western": "west",
        "center": "center", "central": "center",
    }
    for kw in keywords:
        mapped = mapping.get(kw)
        if mapped and mapped in zones:
            return True
    return len(keywords) == 0


def extract_subject_near(question):
    patterns = [
        r"near\s+(.+?)(?:\?|$)",
        r"close to\s+(.+?)(?:\?|$)",
        r"adjacent to\s+(.+?)(?:\?|$)",
        r"nearby\s+(.+?)(?:\?|$)",
        r"nearest to\s+(.+?)(?:\?|$)",
        r"next to\s+(.+?)(?:\?|$)",
    ]
    for pat in patterns:
        m = re.search(pat, question, re.IGNORECASE)
        if m:
            return m.group(1).strip().rstrip("?.,")
    return None


# ═══════════════════════════════════════════════════════════════════════════════
# VLM FALLBACK
# ═══════════════════════════════════════════════════════════════════════════════

def load_vlm():
    from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
    local_72b = Path("./model_weights/Qwen2-VL-72B-Instruct-AWQ")
    local_7b = Path("./model_weights/Qwen2-VL-7B-Instruct")
    if local_72b.exists():
        model_name = str(local_72b)
    elif local_7b.exists():
        model_name = str(local_7b)
    else:
        raise FileNotFoundError("Model weights not found. Run setup.bash first.")

    print(f"Loading VLM: {model_name}")
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(model_name)
    return model, processor


def answer_with_vlm(model, processor, pil_image, question, options, ocr_context=""):
    from qwen_vl_utils import process_vision_info

    opt_str = "\n".join(f"{i+1}. {opt}" for i, opt in enumerate(options))

    ocr_hint = ""
    if ocr_context:
        ocr_hint = (
            f"\n\nHere are some text labels detected on the map for reference:\n{ocr_context}\n"
        )

    prompt = (
        f"This is a detailed geographic map showing streets, water bodies, landmarks, and labeled locations. "
        f"Read all the text labels carefully including small ones.{ocr_hint}\n\n"
        f"Question: {question}\n\n"
        f"Options:\n{opt_str}\n\n"
        f"Instructions: Based on the map labels and features visible in the image, pick the best answer. "
        f"Respond with ONLY a single digit: 1, 2, 3, or 4. "
        f"If you cannot determine the answer, respond with 5."
    )

    messages = [{"role": "user", "content": [
        {"type": "image", "image": pil_image},
        {"type": "text", "text": prompt},
    ]}]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text], images=image_inputs, videos=video_inputs,
        padding=True, return_tensors="pt",
    ).to(model.device)

    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=16, do_sample=False)

    trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, generated_ids)]
    output_text = processor.batch_decode(trimmed, skip_special_tokens=True,
                                          clean_up_tokenization_spaces=False)[0].strip()

    answer = 5
    for ch in output_text:
        if ch in "12345":
            answer = int(ch)
            break
    return answer, output_text


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_dir", required=True, help="Path to test directory")
    args = parser.parse_args()

    test_dir = Path(args.test_dir)
    patches_dir = test_dir / "patches"
    test_csv = test_dir / "test.csv"

    # ── Step 1: Stitch map ────────────────────────────────────────────────────
    print("Loading patches...")
    patches = load_patches(str(patches_dir))
    n_patches = len(patches)
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
    print(f"  Saved: {map_path} ({full_map.shape})")

    # ── Step 2: OCR text extraction ───────────────────────────────────────────
    print("Extracting text from map with OCR...")
    ocr_texts, map_h, map_w = extract_text_from_map(map_path)
    print(f"  Found {len(ocr_texts)} text labels")
    for t in ocr_texts[:10]:
        print(f"    '{t['text']}' at ({t['cx']:.0f}, {t['cy']:.0f}) conf={t['confidence']:.2f}")
    if len(ocr_texts) > 10:
        print(f"    ... and {len(ocr_texts) - 10} more")

    # ── Step 3: Load VLM (for fallback) ───────────────────────────────────────
    print("Loading VLM for fallback...")
    try:
        vlm_model, vlm_processor = load_vlm()
        vlm_available = True
    except (FileNotFoundError, ImportError, Exception) as e:
        print(f"  VLM not available: {e}")
        vlm_available = False

    pil_image = Image.open(map_path).convert("RGB") if vlm_available else None

    # Build OCR context string for VLM prompts
    ocr_context_str = ", ".join(
        t["text"] for t in sorted(ocr_texts, key=lambda x: -x["confidence"])[:60]
    )

    # ── Step 4: Answer questions ──────────────────────────────────────────────
    print("Answering questions...")
    df = pd.read_csv(str(test_csv))
    results = []

    for _, row in df.iterrows():
        qid = row["id"]
        question = row["question"]
        options = [row["option_1"], row["option_2"], row["option_3"], row["option_4"]]

        # Try OCR first
        ocr_answer, ocr_conf = answer_with_ocr(question, options, ocr_texts, map_h, map_w)

        if ocr_answer is not None and ocr_conf >= 0.6:
            answer = ocr_answer
            method = f"OCR (conf={ocr_conf:.2f})"
        elif vlm_available:
            # VLM fallback with OCR context hints
            answer, raw = answer_with_vlm(
                vlm_model, vlm_processor, pil_image, question, options, ocr_context_str
            )
            method = f"VLM (raw={raw!r})"
        elif ocr_answer is not None:
            answer = ocr_answer
            method = f"OCR-low (conf={ocr_conf:.2f})"
        else:
            answer = 5
            method = "SKIP"

        print(f"  {qid}: {answer} [{method}]")
        print(f"    Q: {question}")
        results.append({"id": qid, "question_num": qid, "option": answer})

    # ── Step 5: Write submission ──────────────────────────────────────────────
    print("Writing submission.csv...")
    with open("submission.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "question_num", "option"])
        writer.writeheader()
        writer.writerows(results)

    print(f"Done. submission.csv written with {len(results)} rows.")


if __name__ == "__main__":
    main()
