# tasp/canonical_mapper.py
import argparse, json, re, yaml, inflect, numpy as np, pandas as pd
from pathlib import Path
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

def load_jsonl(p):
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            line=line.strip()
            if line: yield json.loads(line)

def normalize(s: str) -> str:
    s = s.lower().strip()
    s = s.replace("-", " ").replace("_"," ")
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s

def singularize(s: str, engine) -> str:
    # naive singularization per token
    toks = s.split()
    out=[]
    for t in toks:
        sng = engine.singular_noun(t)
        out.append(sng if sng else t)
    return " ".join(out)

def compile_regex_map(regex_rules):
    comp={}
    for canon, patterns in (regex_rules or {}).items():
        comp[canon] = [re.compile(p, flags=re.I) for p in patterns]
    return comp

def build_synonym_map(synonyms):
    m={}
    for canon, syns in (synonyms or {}).items():
        for s in syns:
            m[normalize(s)] = canon
    return m

def embed_labels(model_name, labels):
    m = SentenceTransformer(model_name)
    embs = m.encode(labels, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)
    return m, embs

def map_term(term, regex_map, syn_map, canon_labels, model, canon_embs, threshold):
    raw = term
    t = normalize(term)
    # regex rules first
    for canon, regs in regex_map.items():
        if any(r.search(raw) or r.search(t) for r in regs):
            return canon, f"regex"
    # direct synonym table
    if t in syn_map:
        return syn_map[t], "synonym"
    # embedding fallback (to nearest canonical label)
    vec = model.encode([t], normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)
    sims = cosine_similarity(vec, canon_embs)[0]
    j = int(np.argmax(sims))
    if float(sims[j]) >= threshold:
        return canon_labels[j], f"embed:{sims[j]:.2f}"
    return "other", f"embed_low:{float(sims[j]):.2f}"

def main(a):
    rules = yaml.safe_load(open(a.rules_yaml, "r", encoding="utf-8"))
    canon_labels = rules["canonical_labels"]
    regex_map = compile_regex_map(rules.get("regex_rules", {}))
    syn_map   = build_synonym_map(rules.get("synonyms", {}))

    # Build embedding space once
    model_name = a.model
    model, canon_embs = embed_labels(model_name, [normalize(x) for x in canon_labels])

    engine = inflect.engine()

    Path(a.out_jsonl).parent.mkdir(parents=True, exist_ok=True)
    rows=[]
    with open(a.out_jsonl, "w", encoding="utf-8") as fout:
        for ex in load_jsonl(a.pred_jsonl):
            new_pairs=[]
            for it in ex.get("pairs", []):
                aspect = it.get("aspect","")
                asp_norm = singularize(normalize(aspect), engine)
                canon, reason = map_term(asp_norm, regex_map, syn_map, canon_labels, model, canon_embs, a.threshold)
                new_pairs.append({"aspect": aspect, "sentiment": it.get("sentiment"), "canonical": canon})
                rows.append({"aspect_raw": aspect, "aspect_norm": asp_norm, "canonical": canon, "reason": reason})
            fout.write(json.dumps({"id": ex.get("id"), "text": ex.get("text"), "pairs": new_pairs}, ensure_ascii=False)+"\n")
    # report
    df = pd.DataFrame(rows)
    df.to_csv(a.report_csv, index=False, encoding="utf-8")
    print("Wrote", a.out_jsonl, "and", a.report_csv)

if __name__ == "__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--pred_jsonl", required=True)
    ap.add_argument("--rules_yaml", default="data/canonical_rules.yml")
    ap.add_argument("--out_jsonl", default="outputs/mapping/pairs_canonical.jsonl")
    ap.add_argument("--report_csv", default="outputs/mapping/mapping_report.csv")
    ap.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--threshold", type=float, default=0.62)  # tune 0.58–0.68
    main(ap.parse_args())
