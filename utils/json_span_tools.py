import json, re, math
from typing import List, Dict, Tuple
from dataclasses import dataclass

JSON_ITEM_KEYS = ("aspect", "sentiment")
VALID_POLARITY = {"positive", "negative", "neutral"}

_WHITESPACE_RE = re.compile(r"\s+")
_PUNCT_RE = re.compile(r"[^\w\s]")

def _normalize_space(s: str) -> str:
    return _WHITESPACE_RE.sub(" ", s).strip()

def _normalize_loose(s: str) -> str:
    s = s.lower()
    s = _WHITESPACE_RE.sub(" ", s)
    s = s.strip()
    return s

def _normalize_alnum(s: str) -> str:
    s = s.lower()
    s = _PUNCT_RE.sub("", s)
    s = _WHITESPACE_RE.sub(" ", s).strip()
    return s

def safe_json_parse(raw: str) -> List[Dict[str, str]]:
    """
    Robust JSON fixer:
    - Finds the first and last bracketed JSON array
    - Repairs common trailing commas / quotes
    - Validates schema; drops invalid items
    """
    if not raw:
        return []
    # Heuristic: grab JSON array
    start = raw.find("[")
    end = raw.rfind("]")
    if start == -1 or end == -1 or end <= start:
        # try to coerce a single object into an array
        raw = raw.strip()
        if raw.startswith("{") and raw.endswith("}"):
            raw = "[" + raw + "]"
        else:
            return []
    else:
        raw = raw[start:end+1]

    # Common minor fixes
    raw = raw.replace("\n", " ").replace("\r", " ")
    raw = raw.replace("'", '"')  # single -> double quotes
    raw = re.sub(r",\s*}", "}", raw)  # trailing comma in object
    raw = re.sub(r",\s*]", "]", raw)  # trailing comma in array

    try:
        data = json.loads(raw)
    except Exception:
        # Last-resort: wrap in array if it's a dict-like string
        try:
            data = json.loads("[" + raw + "]")
        except Exception:
            return []

    if not isinstance(data, list):
        data = [data]

    cleaned = []
    for item in data:
        if not isinstance(item, dict):
            continue
        # key normalization (tolerate capitalization)
        norm = {k.lower(): v for k, v in item.items()}
        if not all(k in norm for k in JSON_ITEM_KEYS):
            # allow alternative keys like 'opinion' -> 'aspect' fallback
            if "opinion" in norm and "sentiment" in norm:
                norm["aspect"] = norm.pop("opinion")
            else:
                continue
        aspect = str(norm["aspect"]).strip()
        sent = str(norm["sentiment"]).lower().strip()
        if sent not in VALID_POLARITY:
            # quick mapping
            if sent in {"pos", "positive "}:
                sent = "positive"
            elif sent in {"neg", "negative "}:
                sent = "negative"
            elif sent in {"neu", "neutral "}:
                sent = "neutral"
            else:
                continue
        if aspect == "":
            continue
        cleaned.append({"aspect": aspect, "sentiment": sent})
    return cleaned

def unique_pairs(pairs: List[Dict[str, str]]) -> List[Dict[str, str]]:
    seen = set()
    out = []
    for p in pairs:
        k = (_normalize_loose(p["aspect"]), p["sentiment"])
        if k not in seen:
            seen.add(k)
            out.append(p)
    return out

def find_best_span(aspect_pred: str, sentence: str) -> Tuple[str, float]:
    """
    Return the best exact substring from `sentence` to represent `aspect_pred`.
    Score = Jaccard on character bigrams + bonus for exact substring match ignoring case.
    """
    if not aspect_pred:
        return "", 0.0
    s_norm = sentence
    a = aspect_pred
    # quick exact (case-insensitive) substring
    idx = s_norm.lower().find(a.lower())
    if idx != -1:
        exact = s_norm[idx:idx+len(a)]
        return exact, 1.0

    # fallback: sliding-window fuzzy on alnum-normalized
    sa = _normalize_alnum(a)
    ss = _normalize_alnum(sentence)
    if not sa:
        return "", 0.0

    # windows around length of 'a'
    L = max(2, min(len(ss), max(len(sa)-2, 3)))
    best, best_score = "", 0.0
    for i in range(0, max(1, len(ss) - L + 1)):
        cand = ss[i:i+L]
        # char bigram jaccard
        def bigrams(x):
            return {x[j:j+2] for j in range(len(x)-1)} if len(x) >= 2 else {x}
        A, B = bigrams(sa), bigrams(cand)
        score = len(A & B) / (len(A | B) + 1e-9)
        if score > best_score:
            # map back to original substring by index in original sentence
            # approximate: take indices in original by matching cand's raw text
            raw_cand = cand.replace(" ", "")
            # brutal fallback: just return original aspect text if not found
            best = aspect_pred
            best_score = score
    return best, best_score

NEG_LEX = {
    "no", "not", "never", "hardly", "barely", "scarcely", "n't", "nor",
    "bad", "poor", "terrible", "awful", "worst", "laggy", "slow", "hot", "overheat",
    "broken", "defective", "dead", "cracked", "flicker", "drain", "drains", "drained",
    "expensive", "pricey"
}
POS_LEX = {
    "good", "great", "excellent", "amazing", "awesome", "fantastic", "best",
    "fast", "snappy", "cool", "quiet", "solid", "reliable", "durable",
    "affordable", "cheap", "value"
}

@dataclass
class RectifyConfig:
    window_tokens: int = 8          # around the matched span
    strong_threshold: int = 2       # how many polarity cues to call it "strong"
    flip_only_on_conflict: bool = True

def polarity_rectify(sentence: str, aspect: str, sentiment: str, cfg: RectifyConfig = RectifyConfig()) -> str:
    """
    Conservative rectifier: looks for strong polarity cues near the aspect.
    Flips only on clear contradiction (e.g., predicted positive, strong NEG cues).
    """
    s = sentence
    if aspect:
        # find index of aspect
        idx = s.lower().find(aspect.lower())
        if idx == -1:
            idx = 0
    else:
        idx = 0

    # token window
    toks = re.findall(r"\w+|[^\w\s]", s.lower())
    # find token index closest to char index
    cum = 0; pos = 0
    for i, t in enumerate(toks):
        cum += len(t)
        if cum >= idx:
            pos = i
            break
    lo = max(0, pos - cfg.window_tokens)
    hi = min(len(toks), pos + cfg.window_tokens + 1)
    win = toks[lo:hi]

    neg_hits = sum(1 for t in win if t in NEG_LEX)
    pos_hits = sum(1 for t in win if t in POS_LEX)

    if cfg.flip_only_on_conflict:
        if sentiment == "positive" and neg_hits >= cfg.strong_threshold and neg_hits > pos_hits:
            return "negative"
        if sentiment == "negative" and pos_hits >= cfg.strong_threshold and pos_hits > neg_hits:
            return "positive"
        return sentiment
    else:
        # choose majority if strong
        if max(neg_hits, pos_hits) >= cfg.strong_threshold:
            return "negative" if neg_hits > pos_hits else "positive"
        return sentiment
