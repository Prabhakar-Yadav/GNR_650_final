"""
Evaluation script: scores predictions against ground truth.
Can be used standalone or after running test_vlm.py.

Usage:
  python evaluate.py --predictions predictions.csv --ground_truth test_50_questions.csv
  OR
  python evaluate.py  (uses default files)
"""

import argparse
import pandas as pd
import sys


def score(predictions, ground_truth):
    correct = 0
    incorrect = 0
    unanswered = 0
    hallucinated = 0
    details = []

    for pred, gt in zip(predictions, ground_truth):
        if pred == 5:
            unanswered += 1
            status = "SKIP"
        elif pred in [1, 2, 3, 4]:
            if pred == gt:
                correct += 1
                status = "CORRECT"
            else:
                incorrect += 1
                status = "WRONG"
        else:
            hallucinated += 1
            status = "HALLUCINATED"
        details.append(status)

    final_score = correct - 0.25 * incorrect - 1 * hallucinated
    total = len(predictions)

    return {
        "total": total,
        "correct": correct,
        "incorrect": incorrect,
        "unanswered": unanswered,
        "hallucinated": hallucinated,
        "score": final_score,
        "accuracy": correct / total if total > 0 else 0,
        "details": details,
    }


def print_report(results, predictions, ground_truth, questions=None):
    print(f"\n{'='*60}")
    print(f"  SCORING REPORT ({results['total']} questions)")
    print(f"{'='*60}")
    print(f"  Correct:       {results['correct']:3d}  (+{results['correct']} pts)")
    print(f"  Incorrect:     {results['incorrect']:3d}  (-{0.25*results['incorrect']:.2f} pts)")
    print(f"  Unanswered:    {results['unanswered']:3d}  (0 pts)")
    print(f"  Hallucinated:  {results['hallucinated']:3d}  (-{results['hallucinated']} pts)")
    print(f"  {'─'*40}")
    print(f"  FINAL SCORE:   {results['score']:.2f} / {results['total']}")
    print(f"  ACCURACY:      {100*results['accuracy']:.1f}%")
    print(f"{'='*60}")

    if questions is not None:
        print(f"\n  Detailed breakdown:")
        print(f"  {'ID':<8} {'Pred':>4} {'GT':>4} {'Status':<12} Question")
        print(f"  {'─'*70}")
        for i, (p, g, s) in enumerate(zip(predictions, ground_truth, results['details'])):
            q = questions[i][:45] + "..." if len(questions[i]) > 45 else questions[i]
            marker = "✓" if s == "CORRECT" else "✗" if s == "WRONG" else "–"
            print(f"  {f'ques_{i+1}':<8} {p:>4} {g:>4} {marker} {s:<10} {q}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", default=None, help="CSV with predictions (id, question_num, option)")
    parser.add_argument("--ground_truth", default="test_50_questions.csv", help="CSV with ground truth")
    args = parser.parse_args()

    gt_df = pd.read_csv(args.ground_truth)
    ground_truth = gt_df["correct_answer"].tolist()
    questions = gt_df["question"].tolist()

    if args.predictions:
        pred_df = pd.read_csv(args.predictions)
        predictions = pred_df["option"].tolist()
    else:
        print("No predictions file provided. Using all-5 (unanswered) baseline.")
        predictions = [5] * len(ground_truth)

    results = score(predictions, ground_truth)
    print_report(results, predictions, ground_truth, questions)
