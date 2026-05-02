"""
Test script to evaluate VLM accuracy on 50 map questions.
Run this on a machine with GPU and model weights available.
Usage: python test_vlm.py
"""

import os
import csv
import time
import torch
import pandas as pd
from pathlib import Path
from PIL import Image


def answer_questions(map_image_path, test_csv_path, model_name=None):
    from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
    from qwen_vl_utils import process_vision_info

    local_weights = Path("./model_weights/Qwen2-VL-7B-Instruct")
    if model_name is None:
        if local_weights.exists():
            model_name = str(local_weights)
        else:
            model_name = "Qwen/Qwen2-VL-7B-Instruct"

    print(f"Loading model: {model_name}")
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(model_name)

    df = pd.read_csv(test_csv_path)
    pil_image = Image.open(map_image_path).convert("RGB")

    predictions = []

    for _, row in df.iterrows():
        qid = row["id"]
        question = row["question"]
        opt1 = row["option_1"]
        opt2 = row["option_2"]
        opt3 = row["option_3"]
        opt4 = row["option_4"]

        prompt = (
            f"You are analyzing a detailed map image. Answer the following multiple choice question.\n\n"
            f"Question: {question}\n\n"
            f"Options:\n"
            f"1. {opt1}\n"
            f"2. {opt2}\n"
            f"3. {opt3}\n"
            f"4. {opt4}\n\n"
            f"Look carefully at all text labels, landmarks, roads, water bodies, and geographic features on the map. "
            f"Respond with ONLY the number (1, 2, 3, or 4) of the correct answer. "
            f"If you are not confident, respond with 5."
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
            generated_ids = model.generate(**inputs, max_new_tokens=16, do_sample=False)

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

        predictions.append(answer)
        print(f"  {qid}: raw={output_text!r} -> {answer}")

    return predictions


def score_predictions(predictions, ground_truth):
    correct = 0
    incorrect = 0
    unanswered = 0
    hallucinated = 0

    for pred, gt in zip(predictions, ground_truth):
        if pred == 5:
            unanswered += 1
        elif pred in [1, 2, 3, 4]:
            if pred == gt:
                correct += 1
            else:
                incorrect += 1
        else:
            hallucinated += 1

    score = correct - 0.25 * incorrect - 1 * hallucinated
    total = len(predictions)

    print(f"\n{'='*50}")
    print(f"SCORING RESULTS ({total} questions)")
    print(f"{'='*50}")
    print(f"  Correct:      {correct}/{total} (+{correct} points)")
    print(f"  Incorrect:    {incorrect}/{total} (-{0.25*incorrect:.2f} points)")
    print(f"  Unanswered:   {unanswered}/{total} (0 points)")
    print(f"  Hallucinated: {hallucinated}/{total} (-{hallucinated} points)")
    print(f"  FINAL SCORE:  {score:.2f}/{total}")
    print(f"  Accuracy:     {100*correct/total:.1f}%")
    print(f"{'='*50}")

    return score


if __name__ == "__main__":
    map_path = "stitched_map.png"
    test_csv = "test_50_questions.csv"

    if not os.path.exists(map_path):
        print("ERROR: stitched_map.png not found. Run inference.py first to generate it.")
        exit(1)

    print("Testing VLM on 50 questions...")
    t0 = time.time()

    df = pd.read_csv(test_csv)
    ground_truth = df["correct_answer"].tolist()

    predictions = answer_questions(map_path, test_csv)

    print(f"\nTotal inference time: {time.time()-t0:.1f}s")

    score = score_predictions(predictions, ground_truth)

    # Save detailed results
    results_df = pd.DataFrame({
        "id": df["id"],
        "question": df["question"],
        "prediction": predictions,
        "ground_truth": ground_truth,
        "correct": [p == g for p, g in zip(predictions, ground_truth)],
    })
    results_df.to_csv("test_results.csv", index=False)
    print(f"\nDetailed results saved to test_results.csv")
