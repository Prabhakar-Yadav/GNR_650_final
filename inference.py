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
    for n_div, scale in [(3, 2.0), (4, 3.0)]:
        step_x = w // n_div
        step_y = h // n_div
        pad_x = step_x // 3
        pad_y = step_y // 3
        for gy in range(n_div):
            for gx in range(n_div):
                x1 = max(0, gx * step_x - pad_x)
                y1 = max(0, gy * step_y - pad_y)
                x2 = min(w, (gx + 1) * step_x + pad_x)
                y2 = min(h, (gy + 1) * step_y + pad_y)
                crop = img[y1:y2, x1:x2]
                s = int(scale)
                upscaled = cv2.resize(crop, (crop.shape[1] * s, crop.shape[0] * s),
                                      interpolation=cv2.INTER_CUBIC)
                run_ocr_on_image(upscaled, offset_x=x1, offset_y=y1, scale=scale)

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

    # Merge horizontally adjacent short tokens into compound labels
    deduped = merge_adjacent_labels(deduped)

    return deduped, h, w


def merge_adjacent_labels(texts):
    """Merge OCR tokens that are spatially adjacent into compound labels."""
    if not texts:
        return texts
    merged_flags = [False] * len(texts)
    result = []
    texts_sorted = sorted(texts, key=lambda t: (round(t["cy"] / 15), t["cx"]))
    for i, t in enumerate(texts_sorted):
        if merged_flags[i]:
            continue
        best_j = None
        best_dx = float("inf")
        avg_char_w = max(8, len(t["text"]) and (20))
        for j, t2 in enumerate(texts_sorted):
            if i == j or merged_flags[j]:
                continue
            dy = abs(t2["cy"] - t["cy"])
            dx = t2["cx"] - t["cx"]
            if dy < 12 and 0 < dx < avg_char_w * (len(t["text"]) + 2) and dx < best_dx:
                best_dx = dx
                best_j = j
        if best_j is not None:
            t2 = texts_sorted[best_j]
            compound_text = t["text"] + " " + t2["text"]
            merged = {
                "text": compound_text,
                "confidence": min(t["confidence"], t2["confidence"]),
                "cx": (t["cx"] + t2["cx"]) / 2,
                "cy": (t["cy"] + t2["cy"]) / 2,
                "norm_x": (t["norm_x"] + t2["norm_x"]) / 2,
                "norm_y": (t["norm_y"] + t2["norm_y"]) / 2,
            }
            result.append(merged)
            merged_flags[best_j] = True
        else:
            result.append(t)
    return result


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
    dx = target["cx"] - ref["cx"]
    dy = target["cy"] - ref["cy"]  # positive = down = south in image coords

    # Precise bearing via atan2
    angle = math.degrees(math.atan2(dy, dx))
    if -22.5 <= angle < 22.5:
        computed_dir = "East"
    elif 22.5 <= angle < 67.5:
        computed_dir = "South-East"
    elif 67.5 <= angle < 112.5:
        computed_dir = "South"
    elif 112.5 <= angle < 157.5:
        computed_dir = "South-West"
    elif angle >= 157.5 or angle < -157.5:
        computed_dir = "West"
    elif -157.5 <= angle < -112.5:
        computed_dir = "North-West"
    elif -112.5 <= angle < -67.5:
        computed_dir = "North"
    else:
        computed_dir = "North-East"

    best_i, best_score = None, 0
    for i, opt in enumerate(options):
        opt_l = opt.lower()
        score = fuzzy_ratio(computed_dir.lower(), opt_l)
        if computed_dir.lower() in opt_l or opt_l in computed_dir.lower():
            score = max(score, 0.75)
        parts = computed_dir.lower().split("-")
        part_matches = sum(1 for p in parts if p in opt_l)
        if part_matches:
            score = max(score, 0.5 + 0.2 * part_matches / len(parts))
        if score > best_score:
            best_score = score
            best_i = i + 1
    if best_i is not None and best_score >= 0.4:
        return best_i, 0.82
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

def should_defer_ocr_to_vlm(question):
    """Return True for questions where OCR text matches are easy to over-trust."""
    q = f" {question.lower()} "
    risky_patterns = [
        " east of ", " west of ", " north of ", " south of ",
        " eastern ", " western ", " northern ", " southern ",
        " north-west", " north west", " south-west", " south west",
        " north-east", " north east", " south-east", " south east",
        " top-left", " top left", " top-right", " top right",
        " bottom-left", " bottom left", " bottom-right", " bottom right",
        " in the north", " in the south", " in the east", " in the west",
        " terrain ", " dense ", " color ", " colored ",
        " pink ", " red ", " infrastructure ", " appears ",
    ]
    if "general direction" in q or " direction of " in q:
        return False
    return any(pattern in q for pattern in risky_patterns)


def get_gpu_memory_gb():
    if torch.cuda.is_available():
        return torch.cuda.get_device_properties(0).total_memory / (1024**3)
    return 0


# Primary model: Qwen2.5-VL-7B-Instruct (no AWQ, loads clean on any GPU >=16GB)
# Fallback: Qwen2-VL-7B-Instruct
MODEL_CANDIDATES = [
    ("Qwen2.5-VL-7B", "Qwen2.5-VL-7B-Instruct", "Qwen/Qwen2.5-VL-7B-Instruct"),
    ("Qwen2-VL-7B",   "Qwen2-VL-7B-Instruct",   "Qwen/Qwen2-VL-7B-Instruct"),
]


def unique_paths(paths):
    seen = set()
    unique = []
    for path in paths:
        resolved = str(path.resolve())
        if resolved not in seen:
            seen.add(resolved)
            unique.append(path)
    return unique


def model_weight_roots():
    script_dir = Path(__file__).resolve().parent
    return unique_paths([script_dir / "model_weights", Path.cwd() / "model_weights"])


def resolve_vl_model_class(model_name):
    normalized = str(model_name).replace("_", "-").lower()
    if "qwen2.5-vl" in normalized:
        from transformers import Qwen2_5_VLForConditionalGeneration
        return Qwen2_5_VLForConditionalGeneration
    from transformers import Qwen2VLForConditionalGeneration
    return Qwen2VLForConditionalGeneration


def load_vlm():
    """Load Qwen2.5-VL-7B-Instruct from local weights (downloaded by setup.bash)."""
    from transformers import AutoProcessor

    gpu_mem = get_gpu_memory_gb()
    print(f"GPU memory: {gpu_mem:.1f} GB")

    roots = model_weight_roots()
    print(f"Model weight roots: {[str(r) for r in roots]}")

    strategies = []
    for desc, local_dir, repo_id in MODEL_CANDIDATES:
        for root in roots:
            candidate = root / local_dir
            if (candidate / "config.json").exists():
                print(f"  Found: {candidate}")
                strategies.append({"name": str(candidate), "desc": f"{desc} (local)", "local_only": True})
                break

    if not strategies:
        raise FileNotFoundError(
            "No local VLM weights found. Run setup.bash first to download the model."
        )

    min_pixels = int(os.environ.get("QWEN_MIN_PIXELS", 256 * 28 * 28))
    max_pixels = int(os.environ.get("QWEN_MAX_PIXELS", 1280 * 28 * 28))

    last_error = None
    for strat in strategies:
        try:
            print(f"Loading: {strat['desc']}...")
            model_cls = resolve_vl_model_class(strat["name"])
            extra_kwargs = {}
            try:
                import flash_attn  # noqa: F401
                extra_kwargs["attn_implementation"] = "flash_attention_2"
            except ImportError:
                pass
            model = model_cls.from_pretrained(
                strat["name"],
                torch_dtype=torch.bfloat16,
                device_map="auto",
                low_cpu_mem_usage=True,
                trust_remote_code=True,
                local_files_only=strat["local_only"],
                **extra_kwargs,
            )
            processor = AutoProcessor.from_pretrained(
                strat["name"],
                trust_remote_code=True,
                local_files_only=strat["local_only"],
                min_pixels=min_pixels,
                max_pixels=max_pixels,
            )
            print(f"VLM loaded: {strat['desc']}")
            return model, processor
        except Exception as e:
            print(f"{strat['desc']} failed: {str(e)[:120]}")
            last_error = e
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    raise RuntimeError(f"All VLM strategies failed. Last: {last_error}")


def get_relevant_crop(pil_image, question, options, ocr_texts, map_h, map_w):
    """Extract a cropped region of the map centered on relevant landmarks."""
    all_locs = []

    for opt in options:
        hits = find_text_on_map(ocr_texts, opt, threshold=0.4)
        if hits:
            all_locs.append((hits[0][0]["cx"], hits[0][0]["cy"]))

    # Extract all capitalized noun phrases from question as landmark candidates
    keywords = re.findall(
        r'(?:near|of|at|in|around|from|between|south of|north of|east of|west of)\s+([A-Z][a-zA-Z\s]+)',
        question
    )
    # Also grab bare proper nouns
    for word in re.findall(r'\b([A-Z][a-zA-Z]{3,}(?:\s+[A-Z][a-zA-Z]+)*)\b', question):
        keywords.append(word)

    for kw in keywords:
        hits = find_text_on_map(ocr_texts, kw.strip(), threshold=0.4)
        if hits:
            all_locs.append((hits[0][0]["cx"], hits[0][0]["cy"]))

    if not all_locs:
        return None

    xs = [loc[0] for loc in all_locs]
    ys = [loc[1] for loc in all_locs]
    cx, cy = np.mean(xs), np.mean(ys)

    # Tight crop when all locs cluster; wider when spread out
    spread = max(
        max(xs) - min(xs) if len(xs) > 1 else 0,
        max(ys) - min(ys) if len(ys) > 1 else 0,
    )
    margin = max(spread * 0.7, max(map_w, map_h) * 0.20)
    margin = min(margin, max(map_w, map_h) * 0.40)

    x1 = max(0, int(cx - margin))
    y1 = max(0, int(cy - margin))
    x2 = min(map_w, int(cx + margin))
    y2 = min(map_h, int(cy + margin))

    if (x2 - x1) < map_w * 0.12 or (y2 - y1) < map_h * 0.12:
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
        generated_ids = model.generate(**inputs, max_new_tokens=64, do_sample=False,
                                       temperature=None, top_p=None)

    trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, generated_ids)]
    return processor.batch_decode(trimmed, skip_special_tokens=True,
                                  clean_up_tokenization_spaces=False)[0].strip()


def parse_vlm_answer(output_text):
    """Parse a 1-4 answer from VLM output text. Returns 5 only as last resort."""
    text = str(output_text).strip()
    if text in {"1", "2", "3", "4"}:
        return int(text)

    answer_patterns = [
        r"(?:answer|option|choice|final)\D{0,24}([1-4])\b",
        r"\b([1-4])\b",
    ]
    for pattern in answer_patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return int(match.group(1))

    option_words = {"one": 1, "two": 2, "three": 3, "four": 4}
    text_lower = text.lower()
    for word, val in option_words.items():
        if word in text_lower:
            return val

    if text == "5":
        return 5
    return 5


def get_question_id(row):
    for col in ("id", "question_id", "question_num"):
        value = row.get(col)
        if value is not None and not pd.isna(value):
            return str(value)
    raise KeyError("test.csv must contain one of: id, question_id, question_num")


def build_ocr_hint(question, options, ocr_texts):
    """Build a compact OCR hint string with per-option spatial locations."""
    if not ocr_texts:
        return ""
    keywords = set()
    for word in re.findall(r'[A-Z][a-zA-Z]+', question):
        keywords.add(word.lower())
    for opt in options:
        for word in re.findall(r'[A-Z][a-zA-Z]+', opt):
            keywords.add(word.lower())

    # Per-option location hints
    opt_locs = []
    for i, opt in enumerate(options):
        hits = find_text_on_map(ocr_texts, opt, threshold=0.4)
        if hits:
            t = hits[0][0]
            zone = "-".join(get_spatial_zone(t["norm_x"], t["norm_y"]))
            pct_x = int(t["norm_x"] * 100)
            pct_y = int(t["norm_y"] * 100)
            opt_locs.append(f"  Option {i+1} ({opt}): found on map at ~{pct_x}% from left, {pct_y}% from top ({zone})")

    # Relevant background labels
    relevant = []
    for t in sorted(ocr_texts, key=lambda x: -x["confidence"])[:120]:
        text_l = t["text"].lower()
        if any(kw in text_l for kw in keywords) or t["confidence"] >= 0.80:
            zone = "-".join(get_spatial_zone(t["norm_x"], t["norm_y"]))
            relevant.append(f'"{t["text"]}" ({zone})')

    hint_parts = []
    if opt_locs:
        hint_parts.append("Option locations on map:\n" + "\n".join(opt_locs))
    if relevant:
        hint_parts.append("Map labels: " + ", ".join(relevant[:50]))
    if hint_parts:
        return "\n\n" + "\n\n".join(hint_parts) + "\n"
    return ""


def answer_with_vlm(model, processor, pil_image, question, options, ocr_context="",
                     ocr_texts=None, map_h=0, map_w=0):
    opt_str = "\n".join(f"{i+1}. {opt}" for i, opt in enumerate(options))
    ocr_hint = build_ocr_hint(question, options, ocr_texts) if ocr_texts else ""

    def make_prompt(image_desc, q, opts_str, hint):
        return (
            f"You are a geographic expert analyzing a detailed map of Mumbai, India. "
            f"The map shows lakes, rivers, roads, buildings, and labeled landmarks. "
            f"{image_desc}{hint}\n"
            f"Question: {q}\n\n"
            f"Options:\n{opts_str}\n\n"
            f"Study the map labels and spatial positions carefully. "
            f"Think step by step: identify where each option is located on the map, "
            f"then pick the one that best answers the question. "
            f"Respond with ONLY the single digit 1, 2, 3, or 4."
        )

    full_prompt = make_prompt("", question, opt_str, ocr_hint)
    full_raw = run_vlm_inference(model, processor, pil_image, full_prompt)
    full_answer = parse_vlm_answer(full_raw)

    # Cropped region pass
    crop_answer = None
    crop_raw = ""
    if ocr_texts and map_h > 0:
        crop = get_relevant_crop(pil_image, question, options, ocr_texts, map_h, map_w)
        if crop is not None:
            crop_prompt = make_prompt(
                "This is a zoomed-in section of the map. ",
                question, opt_str, ocr_hint
            )
            crop_raw = run_vlm_inference(model, processor, crop, crop_prompt)
            crop_answer = parse_vlm_answer(crop_raw)

    # Anti-position-bias: if answer is not 1 and full/crop agree on non-1,
    # do a third pass with options shuffled to verify
    candidates = [a for a in [full_answer, crop_answer] if a is not None and a != 5]
    agreed = full_answer if (crop_answer is not None and full_answer == crop_answer and full_answer != 5) else None

    if agreed is not None:
        if agreed != 1:
            # Shuffle options: put agreed-option first to see if it still wins
            shuffled_map = list(range(len(options)))
            shuffled_map.insert(0, shuffled_map.pop(agreed - 1))
            shuffled_opts = [options[i] for i in shuffled_map]
            shuffled_str = "\n".join(f"{i+1}. {opt}" for i, opt in enumerate(shuffled_opts))
            verify_prompt = make_prompt("", question, shuffled_str, ocr_hint)
            verify_raw = run_vlm_inference(model, processor, pil_image, verify_prompt)
            verify_parsed = parse_vlm_answer(verify_raw)
            if verify_parsed != 5:
                actual_answer = shuffled_map[verify_parsed - 1] + 1
                if actual_answer == agreed:
                    return agreed, f"VLM-verified={full_raw!r}"
                # Disagreement — return majority of three
                votes = {}
                for v in [full_answer, crop_answer if crop_answer else full_answer, actual_answer]:
                    votes[v] = votes.get(v, 0) + 1
                winner = max(votes, key=votes.get)
                return winner, f"VLM-majority={full_raw!r}"
        return agreed, f"VLM-both={full_raw!r}"

    if full_answer == 5 and crop_answer is not None and crop_answer != 5:
        return crop_answer, f"VLM-crop={crop_raw!r}"
    if full_answer != 5:
        return full_answer, f"VLM-full={full_raw!r}"
    if crop_answer is not None and crop_answer != 5:
        return crop_answer, f"VLM-crop={crop_raw!r}"
    return 5, f"VLM-skip full={full_raw!r}"


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_dir", default=None, help="Path to test directory")
    args = parser.parse_args()

    # Support multiple ways to specify test directory
    test_dir = None
    if args.test_dir:
        test_dir = Path(args.test_dir)
    else:
        env_test_dir = os.environ.get('TEST_DIR')
        if env_test_dir:
            test_dir = Path(env_test_dir)
        else:
            # Scan current dir, parent dir, and grandparent dir for any folder
            # that contains both patches/ and test.csv — works regardless of name
            def find_test_dir(search_root):
                search_root = Path(search_root)
                for candidate in search_root.iterdir():
                    if candidate.is_dir():
                        if (candidate / "patches").is_dir() and (candidate / "test.csv").exists():
                            return candidate
                return None

            script_dir = Path(__file__).resolve().parent
            for search_root in [script_dir, script_dir.parent, script_dir.parent.parent, Path.cwd(), Path.cwd().parent]:
                found = find_test_dir(search_root)
                if found:
                    test_dir = found
                    break

    if test_dir is None or not test_dir.exists():
        raise FileNotFoundError(
            "Test directory not found. Specify with: --test_dir <path>\n"
            "Or set TEST_DIR environment variable.\n"
            "Auto-discovery scans for any folder containing patches/ and test.csv."
        )

    patches_dir = test_dir / "patches"
    test_csv = test_dir / "test.csv"

    print(f"Using test directory: {test_dir}")

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
    skip_vlm = os.environ.get('SKIP_VLM', '0') == '1'
    if skip_vlm:
        print("VLM skipped (SKIP_VLM=1)")
        vlm_available = False
    else:
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
        qid = get_question_id(row)
        question = row["question"]
        options = [row["option_1"], row["option_2"], row["option_3"], row["option_4"]]

        # Try OCR first
        ocr_answer, ocr_conf = answer_with_ocr(question, options, ocr_texts, map_h, map_w)
        risky_ocr = should_defer_ocr_to_vlm(question)

        if ocr_answer is not None and ocr_conf >= 0.85 and not risky_ocr:
            answer = ocr_answer
            method = f"OCR (conf={ocr_conf:.2f})"
        elif vlm_available:
            try:
                vlm_ans, raw = answer_with_vlm(
                    vlm_model, vlm_processor, pil_image, question, options,
                    ocr_context_str, ocr_texts, map_h, map_w,
                )
            except Exception as e:
                print(f"    VLM failed on {qid}: {type(e).__name__}: {str(e)[:180]}")
                vlm_available = False
                vlm_ans, raw = 5, f"VLM-error={type(e).__name__}"

            if vlm_ans == 5 and ocr_answer is not None and ocr_conf >= 0.55:
                answer = ocr_answer
                method = f"OCR-fallback (conf={ocr_conf:.2f})"
            elif vlm_ans == 5 and ocr_answer is not None and ocr_conf >= 0.40:
                answer = ocr_answer
                method = f"OCR-emergency (conf={ocr_conf:.2f})"
            elif vlm_ans == 5:
                # Last resort: pick option with best OCR match score
                best_opt, best_sc = None, 0
                for i, opt in enumerate(options):
                    hits = find_text_on_map(ocr_texts, opt, threshold=0.3)
                    if hits and hits[0][1] > best_sc:
                        best_sc = hits[0][1]
                        best_opt = i + 1
                if best_opt is not None and best_sc >= 0.35:
                    answer = best_opt
                    method = f"OCR-best-match (sc={best_sc:.2f})"
                else:
                    answer = 5
                    method = "SKIP"
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
        print(f"    Q: {question[:120]}")
        if "VLM" in method:
            print(f"    Raw: {method}")
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
