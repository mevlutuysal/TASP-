from transformers import AutoTokenizer, AutoModelForTokenClassification, AutoModelForSequenceClassification
import torch, argparse, json
from utils_io import read_jsonl
from utils_tagging import ID2POL

ATE_TAGS = ["O","B-ASP","I-ASP"]

def decode_ate_spans(text, offsets, tag_ids):
    tags = ["O","B-ASP","I-ASP"]
    spans=[]
    i=0
    while i < len(tag_ids):
        if tags[tag_ids[i]] == "B-ASP":
            start = offsets[i][0]
            j = i + 1
            while j < len(tag_ids) and tags[tag_ids[j]] == "I-ASP":
                j += 1
            end = offsets[j-1][1]
            term = text[start:end]
            # trim common punctuation/space at edges
            term = term.strip(" \t\n\r\"'.,;:!?()[]{}")
            if term:
                spans.append((start, end, term))
            i = j
        else:
            i += 1
    return spans


def main(args):
    # load models
    ate_tok = AutoTokenizer.from_pretrained(args.ate_dir)
    ate_mod = AutoModelForTokenClassification.from_pretrained(args.ate_dir).eval().to(args.device)
    atsa_tok = AutoTokenizer.from_pretrained(args.atsa_dir)
    atsa_mod = AutoModelForSequenceClassification.from_pretrained(args.atsa_dir).eval().to(args.device)

    out = []
    with torch.no_grad():
        for ex in read_jsonl(args.input_jsonl):
            enc = ate_tok(ex["text"], return_offsets_mapping=True, return_tensors="pt", truncation=True, max_length=128).to(args.device)
            offs = enc.pop("offset_mapping").squeeze(0).tolist()
            logits = ate_mod(**enc).logits.squeeze(0).cpu()
            pred = logits.argmax(-1).tolist()
            # remove specials (offset (0,0))
            valid = [(i,o) for i,o in enumerate(offs) if not (o[0]==0 and o[1]==0)]
            pred_valid = [pred[i] for i,_ in valid]
            offs_valid = [o for _,o in valid]
            spans = decode_ate_spans(ex["text"], offs_valid, pred_valid)

            pairs=[]
            for (fr,to,term) in spans:
                enc2 = atsa_tok(ex["text"], term, return_tensors="pt", truncation=True, max_length=128).to(args.device)
                logits2 = atsa_mod(**enc2).logits.squeeze(0).cpu()
                pol_id = int(logits2.argmax(-1))
                pairs.append({"aspect": term, "from": fr, "to": to, "sentiment": ID2POL[pol_id]})
            out.append({"id": ex.get("id"), "text": ex["text"], "pairs": pairs})

    with open(args.out_path, "w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False)+"\n")
    print("Wrote", args.out_path)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_jsonl", default="data/semeval14/laptops_test.jsonl")
    ap.add_argument("--ate_dir", default="outputs/ate/deberta_v3_large")
    ap.add_argument("--atsa_dir", default="outputs/atsa/deberta_v3_large")
    ap.add_argument("--out_path", default="outputs/ate/infer_pairs.jsonl")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    main(args)
