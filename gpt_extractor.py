import os, json, ast, argparse, time
from dotenv import load_dotenv
from openai import OpenAI
from utils_io import read_jsonl

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

SYSTEM = (
"You are an information extraction assistant. "
"Extract aspect terms and their sentiment (positive/negative/neutral). "
"Return ONLY a JSON list of objects with keys: aspect, sentiment."
)

USER_TMPL = (
"Sentence: {text}\n"
"Rules:\n"
" - aspect is a span from the sentence (verbatim).\n"
" - sentiment ∈ {{positive, negative, neutral}}.\n"
"Output JSON list only."
)

def safe_list(s):
    try:
        return json.loads(s)
    except Exception:
        try:
            obj = ast.literal_eval(s)
            return obj if isinstance(obj, list) else []
        except Exception:
            return []

def main(args):
    out=[]
    for ex in read_jsonl(args.input_jsonl):
        prompt = USER_TMPL.format(text=ex["text"])
        resp = client.chat.completions.create(
            model=args.model,  # e.g., "gpt-4o-mini" or similar
            messages=[{"role":"system", "content":SYSTEM},
                      {"role":"user",   "content":prompt}],
            temperature=0,
            response_format={"type":"json_object"} if args.strict_json else None
        )
        content = resp.choices[0].message.content
        if args.strict_json:
            # response should be {"items":[{...},...]}
            try:
                obj = json.loads(content)
                items = obj.get("items", [])
            except Exception:
                items = []
        else:
            items = safe_list(content)
        pairs=[]
        for it in items:
            a = (it.get("aspect") or "").strip()
            s = (it.get("sentiment") or "neutral").strip().lower()
            if a:
                if s not in {"positive","negative","neutral"}: s="neutral"
                pairs.append({"aspect":a,"sentiment":s})
        out.append({"id": ex.get("id"), "text": ex["text"], "pairs": pairs})
        time.sleep(args.sleep)
    with open(args.out_path, "w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False)+"\n")
    print("Wrote", args.out_path)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_jsonl", default="data/semeval14/laptops_test.jsonl")
    ap.add_argument("--model", default="gpt-4o-mini")
    ap.add_argument("--out_path", default="outputs/gpt/infer_pairs.jsonl")
    ap.add_argument("--strict_json", action="store_true")
    ap.add_argument("--sleep", type=float, default=0.0)
    args = ap.parse_args()
    main(args)
