import argparse
import csv
import json
import re
from collections import Counter

ASPECT_KEYS = [
    "aspect", "target", "term", "feature", "category",
    "a", "t", "aspect_text", "target_text", "feature_text"
]

WS_RE = re.compile(r"\s+")

def norm_text(s: str) -> str:
    s = s.strip().lower()
    s = WS_RE.sub(" ", s)
    return s

def get_aspect_from_pair(pair):
    """
    Returns aspect string or None from a pair that can be:
    - ["battery", "positive"]  (list/tuple)
    - {"aspect": "battery", "sentiment": "positive"} (dict)
    - {"a": "battery", "s": "pos"} (dict)
    - Nested dicts (we try a couple of obvious nests)
    """
    if pair is None:
        return None

    # list / tuple
    if isinstance(pair, (list, tuple)):
        if len(pair) >= 1 and isinstance(pair[0], str):
            return pair[0]
        return None

    # dict
    if isinstance(pair, dict):
        # direct keys
        for k in ASPECT_KEYS:
            val = pair.get(k)
            if isinstance(val, str) and val.strip():
                return val

        # nested common shapes (very permissive but safe)
        # e.g., {"aspect": {"text": "battery", ...}}
        for k in ASPECT_KEYS:
            v = pair.get(k)
            if isinstance(v, dict):
                for kk in ("text", "span", "value", "name"):
                    vv = v.get(kk)
                    if isinstance(vv, str) and vv.strip():
                        return vv

        # sometimes pair has {"span": {"aspect": "..."}} etc.
        for outer_k in ("span", "node", "term", "target"):
            v = pair.get(outer_k)
            if isinstance(v, dict):
                for k in ASPECT_KEYS + ["text", "value", "name"]:
                    vv = v.get(k)
                    if isinstance(vv, str) and vv.strip():
                        return vv

    # unknown type
    return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_jsonl", required=True, help="Input JSONL with extracted pairs")
    ap.add_argument("--top_k", type=int, default=50, help="Number of top aspects to print")
    ap.add_argument("--out_csv", default=None, help="Optional CSV path to write aspect,freq")
    args = ap.parse_args()

    counter = Counter()
    total_pairs = 0
    bad_pairs = 0
    lines = 0

    with open(args.in_jsonl, "r", encoding="utf8") as f:
        for line in f:
            lines += 1
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                # skip malformed lines
                continue

            # Prefer canonical if present
            pairs = obj.get("pairs_canon") or obj.get("pairs") or []
            if not isinstance(pairs, (list, tuple)):
                # some pipelines output a dict with a key inside; try to unwrap
                if isinstance(pairs, dict):
                    # common anchors: {"items": [...]}, {"data": [...]}
                    for k in ("items", "data", "list", "pairs"):
                        if isinstance(pairs.get(k), list):
                            pairs = pairs[k]
                            break
                else:
                    pairs = []

            for p in pairs:
                asp = get_aspect_from_pair(p)
                if isinstance(asp, str) and asp.strip():
                    counter[norm_text(asp)] += 1
                    total_pairs += 1
                else:
                    bad_pairs += 1

    print(f"[OK] Lines read          : {lines}")
    print(f"[OK] Pairs processed     : {total_pairs}")
    print(f"[OK] Pairs could not parse: {bad_pairs}")
    print(f"[OK] Unique aspects      : {len(counter)}\n")

    for asp, freq in counter.most_common(args.top_k):
        print(f"{asp}\t{freq}")

    if args.out_csv:
        with open(args.out_csv, "w", newline="", encoding="utf8") as wf:
            w = csv.writer(wf)
            w.writerow(["aspect", "freq"])
            for asp, freq in counter.most_common():
                w.writerow([asp, freq])
        print(f"\n[OK] Wrote CSV → {args.out_csv}")

if __name__ == "__main__":
    main()
