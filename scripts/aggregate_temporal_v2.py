
import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple, Any

import pandas as pd
import numpy as np

DATE_KEYS = ["date", "timestamp", "review_time", "review_date", "time"]

def iter_jsonl(path: str) -> Iterable[dict]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception:
                continue

def parse_date(record: dict) -> Optional[pd.Timestamp]:
    for k in DATE_KEYS:
        if k in record and record[k] is not None:
            v = record[k]
            if isinstance(v, (int, float)):
                val = int(v)
                try:
                    if val > 1e12:
                        return pd.to_datetime(val, unit="ms")
                    if val > 1e10:
                        return pd.to_datetime(val, unit="ms")
                    return pd.to_datetime(val, unit="s")
                except Exception:
                    pass
            if isinstance(v, str):
                v = v.strip()
                if not v:
                    continue
                for fmt in (None, "%Y-%m-%d", "%Y/%m/%d", "%d-%m-%Y", "%m/%d/%Y"):
                    try:
                        ts = pd.to_datetime(v, format=fmt) if fmt else pd.to_datetime(v)
                        return ts
                    except Exception:
                        continue
    return None

def normalize_sentiment(s: str) -> Optional[str]:
    if s is None:
        return None
    s = str(s).strip().lower()
    if s in {"pos", "positive", "+", "1"}:
        return "positive"
    if s in {"neg", "negative", "-", "-1"}:
        return "negative"
    if s in {"neu", "neutral", "0"}:
        return "neutral"
    return None

def choose_group_value(rec: dict, level: str) -> str:
    if level == "global":
        return "GLOBAL"
    if level == "brand":
        return (rec.get("brand") or "Unknown").strip() or "Unknown"
    if level == "family":
        return (rec.get("family") or "Unknown").strip() or "Unknown"
    if level == "product":
        for k in ("parent_asin", "asin", "product_id"):
            if rec.get(k):
                return str(rec[k]).strip()
        return "Unknown"
    return "GLOBAL"

def aggregate(
    input_jsonl: str,
    group_level: str,
    min_monthly_mentions: int = 0,
) -> pd.DataFrame:
    rows = []
    n_records = 0
    n_pairs = 0
    n_missing_date = 0

    for rec in iter_jsonl(input_jsonl):
        n_records += 1
        ts = parse_date(rec)
        if ts is None:
            n_missing_date += 1
            continue
        month = pd.Timestamp(year=ts.year, month=ts.month, day=1)

        grp_val = choose_group_value(rec, group_level)

        pairs = rec.get("pairs") or []
        if not isinstance(pairs, list):
            continue

        for p in pairs:
            aspect_raw = (p.get("canonical") or p.get("aspect") or "").strip().lower()
            if not aspect_raw:
                continue
            sent = normalize_sentiment(p.get("sentiment"))
            if sent is None:
                continue
            rows.append({"group": grp_val, "aspect": aspect_raw, "month": month, "sentiment": sent})
            n_pairs += 1

    if not rows:
        raise RuntimeError("No valid rows found. Check that your input has non-empty `pairs` with sentiments and dates.")

    df = pd.DataFrame(rows)
    counts = (
        df.groupby(["group", "aspect", "month", "sentiment"])
          .size()
          .unstack(fill_value=0)
          .reset_index()
          .rename_axis(None, axis=1)
    )

    for col in ["positive", "negative", "neutral"]:
        if col not in counts.columns:
            counts[col] = 0

    counts["mentions"] = counts[["positive", "negative", "neutral"]].sum(axis=1)
    denom = counts["mentions"].replace(0, np.nan)
    counts["ASI"] = (counts["positive"] - counts["negative"]) / denom
    counts["ASI"] = counts["ASI"].fillna(0.0).clip(-1.0, 1.0)
    counts["p_pos"] = counts["positive"] / denom
    counts["p_neg"] = counts["negative"] / denom
    counts["p_neu"] = counts["neutral"] / denom
    counts[["p_pos", "p_neg", "p_neu"]] = counts[["p_pos", "p_neg", "p_neu"]].fillna(0.0)

    if min_monthly_mentions > 0:
        counts = counts[counts["mentions"] >= min_monthly_mentions].copy()

    counts = counts.sort_values(["group", "aspect", "month"]).reset_index(drop=True)

    print(f"[OK] n_records={n_records:,}, n_pairs={n_pairs:,}, missing_date={n_missing_date:,}, out_rows={len(counts):,}")
    return counts

def save_outputs(df: pd.DataFrame, out_csv: Optional[str], out_parquet: Optional[str]) -> None:
    if out_csv:
        Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_csv, index=False)
        print(f"[OK] wrote CSV: {out_csv}")
    if out_parquet:
        Path(out_parquet).parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out_parquet, index=False)
        print(f"[OK] wrote Parquet: {out_parquet}")

def main():
    ap = argparse.ArgumentParser(description="Aggregate monthly ASI with selectable grouping level: global/brand/family/product.")
    ap.add_argument("--input_jsonl", required=True, help="Enriched reviews JSONL with `pairs` extracted.")
    ap.add_argument("--group_level", choices=["global", "brand", "family", "product"], default="brand")
    ap.add_argument("--min_monthly_mentions", type=int, default=30)
    ap.add_argument("--out_csv", default=None)
    ap.add_argument("--out_parquet", default=None)
    args = ap.parse_args()

    df = aggregate(
        input_jsonl=args.input_jsonl,
        group_level=args.group_level,
        min_monthly_mentions=args.min_monthly_mentions,
    )
    save_outputs(df, args.out_csv, args.out_parquet)

if __name__ == "__main__":
    main()
