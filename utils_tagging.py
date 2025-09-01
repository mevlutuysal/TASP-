from typing import List, Tuple

def spans_to_bio(offsets: List[Tuple[int,int]], aspects: List[dict], labels=("B-ASP","I-ASP","O")):
    L = ["O"] * len(offsets)
    for a in aspects:
        fr, to = a["from"], a["to"]
        inside = False
        for i,(s,e) in enumerate(offsets):
            if e<=fr or s>=to:
                continue
            if not inside:
                L[i] = "B-ASP"
                inside = True
            else:
                L[i] = "I-ASP"
    return L

POL2ID = {"positive":0,"negative":1,"neutral":2,"conflict":3}
ID2POL = {v:k for k,v in POL2ID.items()}
