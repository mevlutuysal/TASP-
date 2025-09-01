import json, random, re, pathlib
p = pathlib.Path(r"outputs\e2e_t5\infer_pairs_flan_base.jsonl")
bad, ok = [], 0
with p.open(encoding="utf-8") as f:
    lines = f.readlines()
for i in random.sample(range(len(lines)), min(50, len(lines))):
    try:
        row = json.loads(lines[i])
        arr = row.get("pairs", row if isinstance(row, list) else [])
        if not isinstance(arr, list) or (arr and not isinstance(arr[0], dict)):
            bad.append((i, lines[i][:160]))
        else:
            ok += 1
    except Exception:
        bad.append((i, lines[i][:160]))
print(f"OK={ok}, BAD={len(bad)}")
for idx, snip in bad[:10]:
    print(idx, snip)
