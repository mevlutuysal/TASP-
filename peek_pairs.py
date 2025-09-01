import json, random, argparse

def load(path):
    with open(path,"r",encoding="utf-8") as f:
        return [json.loads(x) for x in f if x.strip()]

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", default="data/semeval14/laptops_test.jsonl")
    ap.add_argument("--pred", default="outputs/ate/infer_pairs.jsonl")
    ap.add_argument("--k", type=int, default=5)
    a = ap.parse_args()

    gold = {ex["id"]: ex for ex in load(a.gold)}
    pred = load(a.pred)
    ids = [ex["id"] for ex in pred if ex.get("id") in gold]
    random.shuffle(ids)
    for sid in ids[:a.k]:
        g, p = gold[sid], next(x for x in pred if x["id"]==sid)
        print("ID:", sid)
        print("TEXT:", g["text"])
        print("GOLD:", [(t["term"], t["polarity"]) for t in g["aspects"]])
        print("PRED:", [(t["aspect"], t["sentiment"]) for t in p["pairs"]])
        print("-"*80)
