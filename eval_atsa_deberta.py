import argparse, json
from sklearn.metrics import accuracy_score, f1_score

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--pred_jsonl")
    args=ap.parse_args()
    y_true, y_pred = [], []
    with open(args.pred_jsonl,"r",encoding="utf-8") as f:
        for line in f:
            ex = json.loads(line)
            y_true.append(ex["label"])
            y_pred.append(ex["pred"])
    acc = accuracy_score(y_true,y_pred)
    f1m = f1_score(y_true,y_pred, average="macro", zero_division=0)
    print(f"Accuracy: {acc:.4f} | Macro-F1: {f1m:.4f}")
