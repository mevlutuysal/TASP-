from __future__ import annotations
from lxml import etree
from pathlib import Path
import json

def parse_semeval14_laptops(xml_path: str):
    tree = etree.parse(xml_path)
    sents = []
    for s in tree.xpath("//sentence"):
        sid = s.get("id")
        txt_el = s.find("text")
        if txt_el is None:
            continue
        text = txt_el.text
        aspects = []
        at = s.find("aspectTerms")
        if at is not None:
            for term in at.findall("aspectTerm"):
                t = term.get("term")
                fr = int(term.get("from"))
                to = int(term.get("to"))
                pol = term.get("polarity")  # positive/negative/neutral/conflict
                aspects.append({"term": t, "from": fr, "to": to, "polarity": pol})
        sents.append({"id": sid, "text": text, "aspects": aspects})
    return sents

def write_jsonl(items, out_path: str):
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    base = Path("data/semeval14")
    train = parse_semeval14_laptops(str(base/"Laptops_Train.xml"))
    test  = parse_semeval14_laptops(str(base/"Laptops_Test_Gold.xml"))
    write_jsonl(train, str(base/"laptops_train.jsonl"))
    write_jsonl(test,  str(base/"laptops_test.jsonl"))
    print("Wrote:", base/"laptops_train.jsonl", base/"laptops_test.jsonl")
