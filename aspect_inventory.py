# tasp/aspect_inventory.py
import argparse, json, collections, pandas as pd
def load_jsonl(p):
    with open(p,"r",encoding="utf-8") as f:
        for line in f:
            line=line.strip()
            if line: yield json.loads(line)

def main(a):
    cnt = collections.Counter()
    for ex in load_jsonl(a.pred_jsonl):
        for it in ex.get("pairs", []):
            t = (it.get("aspect") or "").strip().lower()
            if t: cnt[t]+=1
    df = pd.DataFrame([{"aspect_raw":k,"freq":v} for k,v in cnt.most_common()])
    df.to_csv(a.out_csv, index=False, encoding="utf-8")
    print("Wrote", a.out_csv, "with", len(df), "unique aspects")
if __name__ == "__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--pred_jsonl", required=True)
    ap.add_argument("--out_csv", default="outputs/mapping/aspect_inventory.csv")
    main(ap.parse_args())
