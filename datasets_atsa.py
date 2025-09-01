from datasets import Dataset
from utils_io import read_jsonl
from transformers import AutoTokenizer
from utils_tagging import POL2ID

def make_atsa_pairs(jsonl_path: str):
    texts, aspects, pols = [], [], []
    for ex in read_jsonl(jsonl_path):
        for a in ex["aspects"]:
            texts.append(ex["text"])
            aspects.append(a["term"])
            pols.append(POL2ID.get(a["polarity"], 2)) # default neutral
    return texts, aspects, pols

def load_atsa_dataset(jsonl_path: str, model_name="microsoft/deberta-v3-large", max_len=128):
    tok = AutoTokenizer.from_pretrained(model_name)
    texts, aspects, pols = make_atsa_pairs(jsonl_path)
    def _enc(batch):
        enc = tok(batch["text"], batch["aspect"], truncation=True, padding=False, max_length=max_len)
        enc["labels"] = batch["label"]
        return enc
    ds = Dataset.from_dict({"text":texts,"aspect":aspects,"label":pols})
    ds = ds.map(_enc, batched=True, remove_columns=["text","aspect","label"])
    ds.set_format(type="torch")
    return ds, tok
