import argparse, json, os, torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel
from tqdm import tqdm
from utils.json_span_tools import (
    safe_json_parse, unique_pairs, find_best_span, polarity_rectify
)

def build_prompt(template: str, sentence: str) -> str:
    return template.replace("{{sentence}}", sentence.strip())

def load_lines(jsonl_path):
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_jsonl", required=True)
    ap.add_argument("--base_model", required=True)
    ap.add_argument("--lora_dir", required=True)
    ap.add_argument("--out_path", required=True)
    ap.add_argument("--prompt_path", default="prompts/json_pairs_prompt.txt")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max_new_tokens", type=int, default=96)
    ap.add_argument("--num_beams", type=int, default=6)
    ap.add_argument("--no_repeat_ngram_size", type=int, default=3)
    ap.add_argument("--length_penalty", type=float, default=0.3)
    ap.add_argument("--apply_span_aligner", action="store_true")
    ap.add_argument("--apply_polarity_rectifier", action="store_true")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out_path), exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    base = AutoModelForSeq2SeqLM.from_pretrained(args.base_model, torch_dtype=torch.float16 if args.device=="cuda" else None)
    model = PeftModel.from_pretrained(base, args.lora_dir)
    model.to(args.device)
    model.eval()

    with open(args.prompt_path, "r", encoding="utf-8") as f:
        tmpl = f.read()

    out_f = open(args.out_path, "w", encoding="utf-8")

    for ex in tqdm(load_lines(args.input_jsonl)):
        # Expect fields: {"id": ..., "sentence": "...", ...}
        sent = ex.get("sentence") or ex.get("text") or ex.get("raw", "")
        prompt = build_prompt(tmpl, sent)

        enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256)
        input_ids = enc["input_ids"].to(args.device)
        attn_mask = enc["attention_mask"].to(args.device)

        with torch.no_grad():
            gen = model.generate(
                input_ids=input_ids,
                attention_mask=attn_mask,
                max_new_tokens=args.max_new_tokens,
                num_beams=args.num_beams,
                no_repeat_ngram_size=args.no_repeat_ngram_size,
                length_penalty=args.length_penalty,
                do_sample=False,
                early_stopping=True,
            )

        out_text = tokenizer.decode(gen[0], skip_special_tokens=True)
        pairs = safe_json_parse(out_text)

        # span aligner + dedup + hallucination filter (copy-only, after-the-fact)
        aligned = []
        for p in pairs:
            best, score = find_best_span(p["aspect"], sent)
            # If we couldn't align with decent confidence AND it's not a literal substring, drop it
            if p["aspect"].lower() not in sent.lower() and score < 0.40:
                continue
            asp = best if best else p["aspect"]
            pol = p["sentiment"]
            if args.apply_polarity_rectifier:
                pol = polarity_rectify(sent, asp, pol)
            aligned.append({"aspect": asp, "sentiment": pol})

        aligned = unique_pairs(aligned)

        out = {
            "id": ex.get("id"),
            "sentence": sent,
            "pairs": aligned
        }
        out_f.write(json.dumps(out, ensure_ascii=False) + "\n")

    out_f.close()
    print(f"✓ Wrote: {args.out_path}")

if __name__ == "__main__":
    main()
