# flan_t5_lora_infer.py (few-shot + robust parsing + debug)
from __future__ import annotations
import argparse, json, re, ast, torch, os
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel

def load_jsonl(p):
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)

# --- FEW-SHOT PROMPT (3 demos) ---
FEWSHOT = (
  "Instruction: Extract aspect–sentiment pairs as a JSON array. "
  "Each item must be an object with keys 'aspect' and 'sentiment'. "
  "The 'sentiment' must be one of ['positive','negative','neutral']. "
  "If there are no pairs, return []. Do not add explanations; return only the JSON array.\n"
  "Examples:\n"
  "Sentence: The battery life is amazing but the keyboard is terrible.\n"
  "Output: [{\"aspect\":\"battery life\",\"sentiment\":\"positive\"},{\"aspect\":\"keyboard\",\"sentiment\":\"negative\"}]\n"
  "Sentence: Screen quality is okay.\n"
  "Output: [{\"aspect\":\"screen quality\",\"sentiment\":\"neutral\"}]\n"
  "Sentence: No specific opinion here.\n"
  "Output: []\n"
)



def to_prompt(text: str) -> str:
    return FEWSHOT + f"Sentence: {text}\nOutput:"



SENT_OK = {"positive", "negative", "neutral"}
SENT_CANON = {
    "pos":"positive", "neg":"negative", "neu":"neutral",
    "positif":"positive","negatif":"negative"
}

PAIR_RE = re.compile(
    r'["\']?\s*aspect["\']?\s*[:=]\s*["\']([^"\']+)["\']\s*[,;]\s*["\']?\s*(?:sentiment|polarity)["\']?\s*[:=]\s*["\']([a-zA-Z]+)["\']',
    re.I
)
PAIR_TUP_RE = re.compile(
    r'\(\s*["\']([^"\']+)["\']\s*,\s*["\'](positive|negative|neutral)["\']\s*\)',
    re.I
)

PAIR_COLON_RE = re.compile(r'([^:\[\]\{\}\n;=]+?)\s*[:=-]\s*(positive|negative|neutral)\b', re.I)

POS_LEX = {"good","great","excellent","amazing","love","best","fast","awesome","perfect","super","outstanding","light"}
NEG_LEX = {"bad","terrible","awful","poor","worst","slow","buggy","hate","broken","noisy","hot","heavy","laggy"}
NEU_LEX = {"ok","okay","fine","average","decent","acceptable","mediocre","neutral"}

def norm_sent(s: str) -> str:
    s = (s or "").strip().lower()
    if s in SENT_OK: return s
    if s in SENT_CANON: return SENT_CANON[s]
    if s.startswith("pos"): return "positive"
    if s.startswith("neg"): return "negative"
    if s.startswith("neu"): return "neutral"
    return "neutral"

def guess_sentiment(tokens):
    # very small heuristic used only when sentiment is missing
    t = " ".join(tokens).lower()
    if any(w in t for w in NEG_LEX) or "not very good" in t or "no " in t:
        return "negative"
    if any(w in t for w in POS_LEX):
        return "positive"
    if any(w in t for w in NEU_LEX):
        return "neutral"
    return "neutral"


def _clean_quotes(s: str) -> str:
    return s.replace("’","'").replace("‘","'").replace("“",'"').replace("”",'"')

def coerce_json_like(txt: str):
    """Try hard to coerce model output to a Python object we can interpret."""
    t0 = txt.strip()
    # 1) direct JSON
    try:
        return json.loads(t0)
    except Exception:
        pass
    # 2) bracket slice
    if "[" in t0 and "]" in t0:
        t1 = t0[t0.find("["): t0.rfind("]")+1]
        for cand in (t1, _clean_quotes(t1), _clean_quotes(t1).replace("'", '"')):
            try:
                return json.loads(cand)
            except Exception:
                pass
            try:
                return ast.literal_eval(cand)
            except Exception:
                pass
    # 3) global quote normalization
    t2 = _clean_quotes(t0).replace("'", '"')
    try:
        return json.loads(t2)
    except Exception:
        pass
    # 4) last resort: literal_eval on normalized
    try:
        return ast.literal_eval(_clean_quotes(t0))
    except Exception:
        pass
    return None

def to_pairs(obj):
    pairs = []
    if isinstance(obj, list):
        for it in obj:
            if isinstance(it, dict):
                a = (it.get("aspect") or it.get("term") or "").strip()
                s = norm_sent(it.get("sentiment") or it.get("polarity") or "")
                if a:
                    pairs.append({"aspect": a, "sentiment": s})
            elif isinstance(it, (list, tuple)) and len(it) >= 1:
                maybe_aspect = str(it[0]).strip()
                sent = None
                for tok in it[1:]:
                    tok_low = str(tok).strip().lower()
                    if tok_low in SENT_OK:
                        sent = tok_low
                        break
                if sent is None:
                    sent = guess_sentiment([str(x) for x in it])
                if maybe_aspect:
                    pairs.append({"aspect": maybe_aspect, "sentiment": sent})
    elif isinstance(obj, dict):
        if "pairs" in obj and isinstance(obj["pairs"], list):
            for it in obj["pairs"]:
                if isinstance(it, dict):
                    a = (it.get("aspect") or it.get("term") or "").strip()
                    s = norm_sent(it.get("sentiment") or it.get("polarity") or "")
                    if a:
                        pairs.append({"aspect": a, "sentiment": s})
        else:
            for k, v in obj.items():
                a = str(k).strip()
                s = norm_sent(str(v))
                if a:
                    pairs.append({"aspect": a, "sentiment": s})
    return pairs


def parse_pairs(txt: str):
    # 1) structured
    obj = coerce_json_like(txt)
    pairs = to_pairs(obj) if obj is not None else []
    if pairs: return pairs

    # 2) tuple-style fallback: ("battery life","positive")
    out = []
    for a, s in PAIR_TUP_RE.findall(txt):
        a = a.strip()
        s = norm_sent(s)
        if a: out.append({"aspect": a, "sentiment": s})
    if out: return out

    out = []
    for a, s in PAIR_COLON_RE.findall(txt):
        a = a.strip()
        s = norm_sent(s)
        if a:
            out.append({"aspect": a, "sentiment": s})
    if out:
        return out

    # 2) regex fallback
    out = []
    for a, s in PAIR_RE.findall(txt):
        a = a.strip(); s = norm_sent(s)
        if a: out.append({"aspect": a, "sentiment": s})
    return out



def main(a):
    # Data
    examples = list(load_jsonl(a.input_jsonl))
    prompts = [to_prompt(ex["text"]) for ex in examples]
    ds = Dataset.from_dict({"prompt": prompts})

    # Model
    tok = AutoTokenizer.from_pretrained(a.base_model)
    base = AutoModelForSeq2SeqLM.from_pretrained(a.base_model)
    model = PeftModel.from_pretrained(base, a.lora_dir)
    # tok = AutoTokenizer.from_pretrained(a.merged_model_dir)
    # model = AutoModelForSeq2SeqLM.from_pretrained(a.merged_model_dir)

    device = torch.device(a.device) if a.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device); model.eval()

    # Tokenize prompts
    def _enc(batch):
        return tok(batch["prompt"], truncation=True, max_length=a.max_src_len, padding=False)
    ds = ds.map(_enc, batched=True, remove_columns=["prompt"])

    # Generate in mini-batches (pad per batch)
    bs = a.batch_size
    decoded = []
    for i in range(0, len(ds), bs):
        chunk = ds[i:i+bs]
        padded = tok.pad(
            {"input_ids": chunk["input_ids"], "attention_mask": chunk["attention_mask"]},
            padding=True, return_tensors="pt"
        )
        input_ids = padded["input_ids"].to(device)
        attn_mask = padded["attention_mask"].to(device)
        # flan_t5_lora_infer.py (Alternative)

        with torch.no_grad():
            gen = model.generate(
                input_ids=input_ids,
                attention_mask=attn_mask,
                max_new_tokens=a.max_new_tokens,
                num_beams=a.num_beams,
                do_sample=a.do_sample,
                temperature=a.temperature,
                top_p=a.top_p
            )

        decoded.extend(tok.batch_decode(gen, skip_special_tokens=True))

    # DEBUG: write a small sample of raw decodes
    dbg_path = os.path.join(os.path.dirname(a.out_path), "debug_decoded.txt")
    try:
        with open(dbg_path, "w", encoding="utf-8") as dbg:
            for t in decoded[:25]:
                dbg.write(t.replace("\n"," ") + "\n")
    except Exception:
        pass

    # Parse and write final jsonl
    non_empty = 0
    with open(a.out_path, "w", encoding="utf-8") as fout:
        for ex, txt in zip(examples, decoded):
            pairs = parse_pairs(txt)
            if pairs: non_empty += 1
            fout.write(json.dumps({"id": ex.get("id"), "text": ex.get("text"), "pairs": pairs}, ensure_ascii=False) + "\n")
    print(f"Wrote {a.out_path} | non-empty examples: {non_empty}/{len(examples)}")
    print(f"(Raw decode sample in: {dbg_path})")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_jsonl", required=True)
    ap.add_argument("--base_model", required=True)
    ap.add_argument("--lora_dir", required=True)
    ap.add_argument("--out_path", required=True)
    ap.add_argument("--device", default=None)
    ap.add_argument("--max_src_len", type=int, default=256)
    ap.add_argument("--max_new_tokens", type=int, default=64)  # tighter helps JSON adherence
    ap.add_argument("--num_beams", type=int, default=1)        # greedy is often best for strict format
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--do_sample", action="store_true")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top_p", type=float, default=0.95)

    args = ap.parse_args()
    main(args)
