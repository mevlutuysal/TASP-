import argparse, json
from utils_metrics import bio_prf1

def read_tags(path):
    y_true, y_pred = [], []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            ex = json.loads(line)
            y_true.append(ex["true_tags"])
            y_pred.append(ex["pred_tags"])
    return y_true, y_pred

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--pred_jsonl")
    args=ap.parse_args()
    y_true, y_pred = read_tags(args.pred_jsonl)
    p,r,f = bio_prf1(y_true,y_pred)
    print(f"Precision: {p:.4f} | Recall: {r:.4f} | F1: {f:.4f}")
