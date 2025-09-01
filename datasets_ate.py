# datasets_ate.py — FIXED

from datasets import Dataset
from transformers import AutoTokenizer

LABEL2ID = {"O":0,"B-ASP":1,"I-ASP":2}
ID2LABEL = {v:k for k,v in LABEL2ID.items()}

def spans_to_bio(offsets, aspects):
    """
    Build BIO over *all* offsets (including specials). Specials have (0,0) → we keep 'O' there.
    """
    L = ["O"] * len(offsets)
    for a in aspects:
        fr, to = a["from"], a["to"]
        inside = False
        for i, (s, e) in enumerate(offsets):
            if s == 0 and e == 0:  # special token (CLS/SEP/etc.)
                continue
            if e <= fr or s >= to:
                continue
            if not inside:
                L[i] = "B-ASP"
                inside = True
            else:
                L[i] = "I-ASP"
    return L

def load_ate_dataset(jsonl_path: str, model_name="microsoft/deberta-v3-large", max_len=128):
    import json

    tok = AutoTokenizer.from_pretrained(model_name)
    texts, aspects = [], []

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            ex = json.loads(line)
            texts.append(ex["text"])
            aspects.append(ex["aspects"])

    def _encode(batch):
        enc = tok(batch["text"], truncation=True, padding=False, max_length=max_len,
                  return_offsets_mapping=True)
        # Make labels same length as offsets
        all_labels = []
        for i, offs in enumerate(enc["offset_mapping"]):
            labs = spans_to_bio(offs, aspects[batch["idx"][i]])
            all_labels.append(labs)
        enc["labels"] = all_labels
        return enc

    ds = Dataset.from_dict({"text": texts, "idx": list(range(len(texts)))})
    ds = ds.map(_encode, batched=True, remove_columns=["text","idx"])

    def _align(batch):
        # Convert string labels to ids; keep -100 on specials
        new_labels = []
        for labs, offs in zip(batch["labels"], batch["offset_mapping"]):
            out = []
            for idx, (s,e) in enumerate(offs):
                if s == 0 and e == 0:
                    out.append(-100)  # special token
                else:
                    out.append(LABEL2ID[labs[idx]])
            new_labels.append(out)
        batch["labels"] = new_labels
        return batch

    ds = ds.map(_align, batched=True)
    ds = ds.remove_columns(["offset_mapping"])
    ds.set_format(type="torch")
    return ds, tok, LABEL2ID, ID2LABEL
