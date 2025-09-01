# eval_pairs_semeval.py
import json, argparse

def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)

def norm_pol(p):
    p = (p or "").lower().strip()
    if p not in {"positive","negative","neutral","conflict"}:
        p = "neutral"
    # default policy: fold 'conflict' into neutral for pair scoring fairness
    return "neutral" if p == "conflict" else p

def score_aspect_only(gold_terms, pred_terms):
    tp = len(gold_terms & pred_terms)
    fp = len(pred_terms - gold_terms)
    fn = len(gold_terms - pred_terms)
    prec = tp/(tp+fp) if tp+fp else 0.0
    rec  = tp/(tp+fn) if tp+fn else 0.0
    f1   = 2*prec*rec/(prec+rec) if prec+rec else 0.0
    return tp, fp, fn, prec, rec, f1

def score_pairs(gold_pairs, pred_pairs):
    tp = len(gold_pairs & pred_pairs)
    fp = len(pred_pairs - gold_pairs)
    fn = len(gold_pairs - pred_pairs)
    prec = tp/(tp+fp) if tp+fp else 0.0
    rec  = tp/(tp+fn) if tp+fn else 0.0
    f1   = 2*prec*rec/(prec+rec) if prec+rec else 0.0
    return tp, fp, fn, prec, rec, f1

def main(args):
    # load gold
    gold_by_id = {}
    for ex in load_jsonl(args.gold_jsonl):
        # normalize terms to lowercase strings
        terms = set()
        pairs = set()
        for a in ex.get("aspects", []):
            t = (a.get("term") or "").strip().lower()
            if not t:
                continue
            terms.add(t)
            pairs.add((t, norm_pol(a.get("polarity"))))
        gold_by_id[ex.get("id")] = {"terms": terms, "pairs": pairs}

    # load preds
    # expected format: {"id":..., "text":..., "pairs":[{"aspect":..., "sentiment":...}, ...]}
    agg_tp_a = agg_fp_a = agg_fn_a = 0
    agg_tp_p = agg_fp_p = agg_fn_p = 0
    sent_match = 0
    sent_total = 0

    for ex in load_jsonl(args.pred_jsonl):
        gid = ex.get("id")
        g = gold_by_id.get(gid, {"terms": set(), "pairs": set()})
        pred_terms = set()
        pred_pairs = set()
        for it in ex.get("pairs", []):
            a = (it.get("aspect") or "").strip().lower()
            s = norm_pol(it.get("sentiment"))
            if not a:
                continue
            pred_terms.add(a)
            pred_pairs.add((a, s))
        # aspect-only
        tp_a, fp_a, fn_a, prec_a, rec_a, f1_a = score_aspect_only(g["terms"], pred_terms)
        agg_tp_a += tp_a; agg_fp_a += fp_a; agg_fn_a += fn_a
        # pair-level
        tp_p, fp_p, fn_p, prec_p, rec_p, f1_p = score_pairs(g["pairs"], pred_pairs)
        agg_tp_p += tp_p; agg_fp_p += fp_p; agg_fn_p += fn_p
        # sentiment accuracy on matched aspects (intersection by aspect only)
        matched_aspects = g["terms"] & pred_terms
        for a in matched_aspects:
            g_lab = None
            p_lab = None
            # find labels (first occurrence is fine since terms usually unique per sentence)
            for (t,lab) in g["pairs"]:
                if t == a:
                    g_lab = lab; break
            for (t,lab) in pred_pairs:
                if t == a:
                    p_lab = lab; break
            if g_lab is not None and p_lab is not None:
                sent_total += 1
                if g_lab == p_lab:
                    sent_match += 1

    prec_a = agg_tp_a/(agg_tp_a+agg_fp_a) if (agg_tp_a+agg_fp_a) else 0.0
    rec_a  = agg_tp_a/(agg_tp_a+agg_fn_a) if (agg_tp_a+agg_fn_a) else 0.0
    f1_a   = 2*prec_a*rec_a/(prec_a+rec_a) if (prec_a+rec_a) else 0.0

    prec_p = agg_tp_p/(agg_tp_p+agg_fp_p) if (agg_tp_p+agg_fp_p) else 0.0
    rec_p  = agg_tp_p/(agg_tp_p+agg_fn_p) if (agg_tp_p+agg_fn_p) else 0.0
    f1_p   = 2*prec_p*rec_p/(prec_p+rec_p) if (prec_p+rec_p) else 0.0

    sent_acc = (sent_match/sent_total) if sent_total else 0.0

    print("=== Aspect-only (string exact match) ===")
    print(f"TP={agg_tp_a} FP={agg_fp_a} FN={agg_fn_a}")
    print(f"Precision={prec_a:.4f} Recall={rec_a:.4f} F1={f1_a:.4f}")
    print("\n=== Aspect+Sentiment (pair exact match) ===")
    print(f"TP={agg_tp_p} FP={agg_fp_p} FN={agg_fn_p}")
    print(f"Precision={prec_p:.4f} Recall={rec_p:.4f} F1={f1_p:.4f}")
    print("\n=== Sentiment Accuracy on matched aspects ===")
    print(f"Acc={sent_acc:.4f} (on {sent_total} matched aspects)")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold_jsonl", default="data/semeval14/laptops_test.jsonl")
    ap.add_argument("--pred_jsonl", default="outputs/ate/infer_pairs.jsonl")
    args = ap.parse_args()
    main(args)
