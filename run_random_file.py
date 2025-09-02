import json, itertools
from collections import Counter
p="outputs/mapping/pairs_canonical.jsonl"
cats=Counter()
with open(p,"r",encoding="utf-8") as f:
    for line in f:
        ex=json.loads(line)
        for it in ex["pairs"]:
            cats[it["canonical"]]+=1
print(cats.most_common(15))