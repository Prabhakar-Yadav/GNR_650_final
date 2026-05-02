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

    # Run OCR on overlapping 3x3 subregions at 2x scale for small text
    n_div = 3
    step_x = w // n_div
    step_y = h // n_div
    pad_x = step_x // 3  # overlap
    pad_y = step_y // 3
    for gy in range(n_div):
        for gx in range(n_div):
            x1 = max(0, gx * step_x - pad_x)
            y1 = max(0, gy * step_y - pad_y)
            x2 = min(w, (gx + 1) * step_x + pad_x)
            y2 = min(h, (gy + 1) * step_y + pad_y)
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
        if len(text_lower) < 2:
            continue

        if query_lower == text_lower:
            score = 1.0
        elif len(query_lower) >= 3 and len(text_lower) >= 3:
            if query_lower in text_lower:
                score = len(query_lower) / len(text_lower)
                score = max(score, 0.75)
            elif text_lower in query_lower:
                score = len(text_lower) / len(query_lower)
                score = max(score, 0.65)
            else:
                score = fuzzy_ratio(query_lower, text_lower)
        else:
            score = fuzzy_ratio(query_lower, text_lower)

        text_words = set(w for w in text_lower.split() if len(w) > 2)
        if query_words and text_words:
            word_overlap = len(query_words & text_words) / len(query_words)
            if word_overlap > 0.2:
                score = max(score, 0.4 + word_overlap * 0.55)

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

    # Skip questions that require visual attributes OCR can't handle
    skip_patterns = [
        'nature of the terrain',
        'terrain', 'dense', 'densely',
        'color', 'colored', 'pink', 'red',
        'building', 'company',
        'infrastructure',
        'appears',
    ]
    if any(pat in q_lower for pat in skip_patterns):
        return None, 0

    # Get OCR matches for all options
    option_data = []
    for i, opt in enumerate(options):
        matches = find_text_on_map(ocr_texts, opt, threshold=0.45)
        best_score = matches[0][1] if matches else 0
        best_match = matches[0][0] if matches else None
        option_data.append((i + 1, best_score, best_match, opt))

    # ── Strategy 0: Direction questions ("direction of X from Y") ────────────
    dir_answer, dir_conf = try_direction_answer(q_lower, options, ocr_texts)
    if dir_answer is not None:
        return dir_answer, dir_conf

    # ── Strategy 1: "between X and Y" questions ─────────────────────────────
    between_answer, between_conf = try_between_answer(q_lower, option_data, ocr_texts)
    if between_answer is not None:
        return between_answer, between_conf

    # ── Strategy 2: Direct text match with spatial awareness ─────────────────
    option_data_sorted = sorted(option_data, key=lambda x: -x[1])

    if option_data_sorted[0][1] >= 0.50:
        top_idx, top_score, top_match, top_opt = option_data_sorted[0]
        second_score = option_data_sorted[1][1] if len(option_data_sorted) > 1 else 0
        gap = top_score - second_score

        spatial_keywords = extract_spatial_keywords(q_lower)
        if spatial_keywords and top_match:
            zones = get_spatial_zone(top_match["norm_x"], top_match["norm_y"])
            spatial_ok = check_spatial_match(spatial_keywords, zones)
            if not spatial_ok:
                for idx, score, match, opt in option_data_sorted[1:]:
                    if score >= 0.45 and match:
                        zones2 = get_spatial_zone(match["norm_x"], match["norm_y"])
                        if check_spatial_match(spatial_keywords, zones2):
                            return idx, 0.68
                return None, 0

        spatial_bonus = 0
        if spatial_keywords and top_match:
            zones = get_spatial_zone(top_match["norm_x"], top_match["norm_y"])
            if check_spatial_match(spatial_keywords, zones):
                spatial_bonus = 0.06

        if top_score >= 0.95 and gap >= max(0.05, 0.1 - spatial_bonus):
            return top_idx, 0.9
        elif top_score >= 0.85 and gap >= max(0.12, 0.18 - spatial_bonus):
            return top_idx, 0.8
        elif top_score >= 0.75 and gap >= max(0.15, 0.22 - spatial_bonus):
            return top_idx, 0.73
        elif top_score >= 0.70 and gap >= 0.18:
            if spatial_keywords and top_match:
                zones = get_spatial_zone(top_match["norm_x"], top_match["norm_y"])
                if check_spatial_match(spatial_keywords, zones):
                    return top_idx, 0.68
        elif top_score >= 0.65 and gap >= 0.22:
            return top_idx, 0.65

        return None, 0

    # ── Strategy 3: Proximity questions ──────────────────────────────────────
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

    return None, 0


def try_direction_answer(q_lower, options, ocr_texts):
    """Handle 'general direction of X from Y' or 'X is ___ of Y' questions."""
    direction_opts = []
    for i, opt in enumerate(options):
        ol = opt.lower()
        if any(d in ol for d in ["north", "south", "east", "west"]):
            direction_opts.append(i)

    if len(direction_opts) < 2:
        return None, 0

    locations = re.findall(
        r'(?:direction of|direction from)\s+(.+?)\s+(?:from|to)\s+(.+?)[\?]',
        q_lower
    )
    if not locations:
        locations = re.findall(
            r'(.+?)\s+(?:from|relative to)\s+(.+?)[\?]',
            q_lower
        )
    if not locations:
        place_match = re.search(
            r'(?:direction|general direction)\s+of\s+(.+?)\s+from\s+(.+?)[\?]',
            q_lower
        )
        if place_match:
            locations = [(place_match.group(1), place_match.group(2))]

    if not locations:
        return None, 0

    target_name, ref_name = locations[0]
    target_hits = find_text_on_map(ocr_texts, target_name.strip(), threshold=0.4)
    ref_hits = find_text_on_map(ocr_texts, ref_name.strip(), threshold=0.4)

    if not target_hits or not ref_hits:
        return None, 0

    target = target_hits[0][0]
    ref = ref_hits[0][0]
    dx = target["norm_x"] - ref["norm_x"]
    dy = target["norm_y"] - ref["norm_y"]

    computed_dir = ""
    if dy < -0.05:
        computed_dir += "North"
    elif dy > 0.05:
        computed_dir += "South"
    if dx > 0.05:
        computed_dir += "-East" if computed_dir else "East"
    elif dx < -0.05:
        computed_dir += "-West" if computed_dir else "West"
    if not computed_dir:
        computed_dir = "Same area"

    best_i, best_score = None, 0
    for i, opt in enumerate(options):
        score = fuzzy_ratio(computed_dir.lower(), opt.lower())
        if score > best_score:
            best_score = score
            best_i = i + 1
    if best_i is not None and best_score >= 0.4:
        return best_i, 0.75
    return None, 0


def try_between_answer(q_lower, option_data, ocr_texts):
    """Handle 'between X and Y' questions using midpoint proximity."""
    m = re.search(r'between\s+(.+?)\s+and\s+(.+?)[\?\s]', q_lower)
    if not m:
        return None, 0

    loc_a_name = m.group(1).strip()
    loc_b_name = m.group(2).strip()
    a_hits = find_text_on_map(ocr_texts, loc_a_name, threshold=0.4)
    b_hits = find_text_on_map(ocr_texts, loc_b_name, threshold=0.4)
    if not a_hits or not b_hits:
        return None, 0

    a_loc = a_hits[0][0]
    b_loc = b_hits[0][0]
    mid_x = (a_loc["cx"] + b_loc["cx"]) / 2
    mid_y = (a_loc["cy"] + b_loc["cy"]) / 2

    best_opt, min_dist = None, float("inf")
    found_any = False
    for idx, score, match, opt in option_data:
        opt_hits = find_text_on_map(ocr_texts, opt, threshold=0.4)
        if opt_hits:
            found_any = True
            oloc = opt_hits[0][0]
            dist = ((mid_x - oloc["cx"]) ** 2 + (mid_y - oloc["cy"]) ** 2) ** 0.5
            if dist < min_dist:
                min_dist = dist
                best_opt = idx
    if best_opt is not None and found_any:
        return best_opt, 0.75
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

    # Try local weights first, then download from HuggingFace on-demand
    model_options = [
        ("./model_weights/Qwen2-VL-72B-Instruct-AWQ", "72B-AWQ (local)"),
        ("./model_weights/Qwen2-VL-7B-Instruct", "7B (local)"),
        ("Qwen/Qwen2-VL-7B-Instruct", "7B (HuggingFace - downloading)"),
    ]

    last_error = None
    for model_name, desc in model_options:
        try:
            print(f"Loading VLM: {desc}...")
            model = Qwen2VLForConditionalGeneration.from_pretrained(
                model_name,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                low_cpu_mem_usage=True,
                trust_remote_code=True,
            )
            processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
            print(f"✓ VLM loaded: {desc}")
            return model, processor
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"⚠ {desc} too large for GPU, trying CPU...")
                try:
                    model = Qwen2VLForConditionalGeneration.from_pretrained(
                        model_name,
                        torch_dtype=torch.float32,
                        device_map="cpu",
                        trust_remote_code=True,
                    )
                    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
                    print(f"✓ VLM loaded on CPU: {desc}")
                    return model, processor
                except Exception:
                    last_error = e
                    continue
            else:
                last_error = e
                continue
        except Exception as e:
            last_error = e
            continue

    if last_error:
        raise RuntimeError(f"VLM loading failed: {last_error}")
    raise FileNotFoundError("Could not load VLM model")


def get_relevant_crop(pil_image, question, options, ocr_texts, map_h, map_w):
    """Extract a cropped region of the map relevant to the question for better VLM accuracy."""
    all_locs = []

    for opt in options:
        hits = find_text_on_map(ocr_texts, opt, threshold=0.4)
        if hits:
            all_locs.append((hits[0][0]["cx"], hits[0][0]["cy"]))

    keywords = re.findall(
        r'(?:near|of|at|in|around|from|south of|north of|east of|west of)\s+([A-Z][a-zA-Z\s]+)',
        question
    )
    for kw in keywords:
        hits = find_text_on_map(ocr_texts, kw.strip(), threshold=0.4)
        if hits:
            all_locs.append((hits[0][0]["cx"], hits[0][0]["cy"]))

    if not all_locs:
        return None

    xs = [loc[0] for loc in all_locs]
    ys = [loc[1] for loc in all_locs]
    cx, cy = np.mean(xs), np.mean(ys)
    margin = max(map_w, map_h) * 0.25
    x1 = max(0, int(cx - margin))
    y1 = max(0, int(cy - margin))
    x2 = min(map_w, int(cx + margin))
    y2 = min(map_h, int(cy + margin))

    if (x2 - x1) < map_w * 0.15 or (y2 - y1) < map_h * 0.15:
        return None

    return pil_image.crop((x1, y1, x2, y2))


def run_vlm_inference(model, processor, image, prompt_text):
    """Run VLM inference on an image with a text prompt. Returns raw output string."""
    from qwen_vl_utils import process_vision_info

    messages = [{"role": "user", "content": [
        {"type": "image", "image": image},
        {"type": "text", "text": prompt_text},
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
    return processor.batch_decode(trimmed, skip_special_tokens=True,
                                  clean_up_tokenization_spaces=False)[0].strip()


def parse_vlm_answer(output_text):
    """Parse a 1-5 answer from VLM output text."""
    for ch in output_text:
        if ch in "12345":
            return int(ch)
    return 5


def answer_with_vlm(model, processor, pil_image, question, options, ocr_context="",
                     ocr_texts=None, map_h=0, map_w=0):
    opt_str = "\n".join(f"{i+1}. {opt}" for i, opt in enumerate(options))

    ocr_hint = ""
    if ocr_context:
        ocr_hint = (
            f"\n\nText labels detected on the map via OCR:\n{ocr_context}\n"
        )

    prompt = (
        f"You are a geographic map analysis expert. This image shows a detailed map with labeled "
        f"streets, water bodies, landmarks, institutional buildings, and geographic features. "
        f"Pay close attention to ALL text labels on the map, including small ones.{ocr_hint}\n\n"
        f"Question: {question}\n\n"
        f"Options:\n{opt_str}\n\n"
        f"Think about what is visible on the map. Pick the option that best matches what the map shows. "
        f"Respond with ONLY a single digit: 1, 2, 3, or 4. "
        f"If you truly cannot determine the answer, respond with 5."
    )

    # Run on full image
    full_raw = run_vlm_inference(model, processor, pil_image, prompt)
    full_answer = parse_vlm_answer(full_raw)

    # Also run on a cropped region for potentially better accuracy on small text
    crop_answer = None
    if ocr_texts and map_h > 0:
        crop = get_relevant_crop(pil_image, question, options, ocr_texts, map_h, map_w)
        if crop is not None:
            crop_prompt = (
                f"This is a zoomed-in section of a geographic map. Read ALL text labels carefully.{ocr_hint}\n\n"
                f"Question: {question}\n\n"
                f"Options:\n{opt_str}\n\n"
                f"Answer with ONLY a single digit: 1, 2, 3, or 4. If unsure, respond with 5."
            )
            crop_raw = run_vlm_inference(model, processor, crop, crop_prompt)
            crop_answer = parse_vlm_answer(crop_raw)

    # If both agree, high confidence. If they disagree, prefer full image (more context).
    if crop_answer is not None and crop_answer != 5:
        if crop_answer == full_answer:
            return full_answer, f"VLM-both={full_raw!r}"
        if full_answer == 5:
            return crop_answer, f"VLM-crop={crop_raw!r}"

    return full_answer, f"VLM-full={full_raw!r}"


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

    # Build OCR context string with locations for VLM prompts
    sorted_ocr = sorted(ocr_texts, key=lambda x: -x["confidence"])[:80]
    ocr_lines = []
    for t in sorted_ocr:
        zone = get_spatial_zone(t["norm_x"], t["norm_y"])
        zone_str = "-".join(zone)
        ocr_lines.append(f'"{t["text"]}" ({zone_str})')
    ocr_context_str = ", ".join(ocr_lines)

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
            vlm_ans, raw = answer_with_vlm(
                vlm_model, vlm_processor, pil_image, question, options,
                ocr_context_str, ocr_texts, map_h, map_w,
            )
            # If VLM is unsure (answer=5) but OCR has a low-confidence answer, use it
            if vlm_ans == 5 and ocr_answer is not None and ocr_conf >= 0.35:
                answer = ocr_answer
                method = f"OCR-fallback (conf={ocr_conf:.2f})"
            else:
                answer = vlm_ans
                method = f"VLM ({raw})"
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
