# pairs format: [{"aspect": "...", "sentiment": "..."}]
def score_pairs(gold_pairs, pred_pairs):
    gold = {(p["aspect"].strip().lower(), p["sentiment"]) for p in gold_pairs}
    pred = {(p["aspect"].strip().lower(), p["sentiment"]) for p in pred_pairs}
    tp = len(gold & pred)
    fp = len(pred - gold)
    fn = len(gold - pred)
    prec = tp / (tp+fp) if tp+fp else 0.0
    rec  = tp / (tp+fn) if tp+fn else 0.0
    f1   = 2*prec*rec/(prec+rec) if prec+rec else 0.0
    return {"tp":tp,"fp":fp,"fn":fn,"precision":prec,"recall":rec,"f1":f1}
