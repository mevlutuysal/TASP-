# evaluate.py
import json
import argparse
from collections import Counter

def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)

def to_comparable_set(pairs):
    """Converts a list of dicts to a set of tuples for easy comparison."""
    return set(
        (p.get("aspect", "").strip(), p.get("sentiment", "").strip())
        for p in pairs
    )

def main(args):
    gold_examples = list(read_jsonl(args.gold_file))
    pred_examples = list(read_jsonl(args.pred_file))

    if len(gold_examples) != len(pred_examples):
        raise ValueError(
            f"Mismatched number of lines between gold ({len(gold_examples)}) "
            f"and prediction ({len(pred_examples)}) files."
        )

    # True Positives, False Positives, False Negatives
    tp, fp, fn = 0, 0, 0

    for gold, pred in zip(gold_examples, pred_examples):
        # Format the gold standard aspects into the same format
        gold_aspects = []
        for aspect in gold.get("aspects", []):
            polarity = (aspect.get("polarity") or "neutral").lower()
            gold_aspects.append({"aspect": aspect["term"], "sentiment": polarity})

        gold_set = to_comparable_set(gold_aspects)
        pred_set = to_comparable_set(pred.get("pairs", []))

        tp += len(gold_set.intersection(pred_set))
        fp += len(pred_set - gold_set)
        fn += len(gold_set - pred_set)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    print("--- ABSA Evaluation Results ---")
    print(f"True Positives:  {tp}")
    print(f"False Positives: {fp}")
    print(f"False Negatives: {fn}")
    print("-------------------------------")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print("-------------------------------")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate ABSA predictions.")
    parser.add_argument("pred_file", type=str, help="Path to the prediction JSONL file.")
    parser.add_argument("gold_file", type=str, help="Path to the gold standard (ground truth) JSONL file.")
    args = parser.parse_args()
    main(args)