#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
import pandas as pd
import numpy as np

# ------------ Tunable gates ------------
YEARS_START = "2016-01"
YEARS_END   = "2022-12"
MIN_MENTIONS_MONTH = 20          # drop months below this, per (group, aspect)
MIN_MONTHS_AFTER_FILTER = 36     # require at least this many kept months
MAX_CONSEC_GAP = 3               # max allowed consecutive missing months in full grid

# ------------ Column names ------------
GROUP_COL   = "group"
ASPECT_COL  = "aspect"
DATE_COL    = "month"
MENTIONS_COL= "mentions"

# --------------------------------------

def month_grid(start=YEARS_START, end=YEARS_END):
    # Month-start frequency grid
    return pd.date_range(start=start, end=end, freq="MS")

def _parse_month(s):
    # robust month parser: supports "2017-05-01", "1.05.2017", etc.
    # we coerce and keep only year-month
    dt = pd.to_datetime(s, errors="coerce", dayfirst=True)
    # Convert to monthly periods, then to timestamps at the start of each month
    return dt.dt.to_period("M").dt.to_timestamp(how="start") if isinstance(dt, pd.Series) else dt


def longest_consecutive_gap(missing_months_index, full_grid):
    if len(missing_months_index) == 0:
        return 0
    pos = pd.Index(full_grid).get_indexer(missing_months_index)
    if len(pos) == 0:
        return 0
    pos = np.sort(pos)
    # split where gaps are not consecutive
    splits = np.where(np.diff(pos) != 1)[0] + 1
    runs = np.split(pos, splits)
    return max((len(r) for r in runs), default=0)


def audit_one_layer(df, layer_name):
    # Normalize/parse date
    df = df.copy()
    if DATE_COL not in df.columns:
        raise ValueError(f"Input is missing '{DATE_COL}' column.")
    if GROUP_COL not in df.columns or ASPECT_COL not in df.columns:
        raise ValueError(f"Input must contain '{GROUP_COL}' and '{ASPECT_COL}'.")

    # Parse dates robustly
    df[DATE_COL] = _parse_month(df[DATE_COL])

    # Keep only rows within the target window
    full_grid = month_grid()
    df = df[df[DATE_COL].between(full_grid.min(), full_grid.max())]

    # Ensure mentions column
    if MENTIONS_COL not in df.columns:
        raise ValueError(f"Input is missing '{MENTIONS_COL}' column.")

    # Prepare audit rows
    # Prepare audit rows
    audit_rows = []
    months_full = len(full_grid)

    # iterate each (group, aspect)
    for (g, a), d in df.groupby([GROUP_COL, ASPECT_COL], sort=False):
        # Month-level rollup (some files may have duplicates per month; keep last)
        d = d.sort_values(DATE_COL).drop_duplicates(subset=[DATE_COL], keep="last")

        # align to full month grid so missing months are visible (NaN)
        d = d.set_index(DATE_COL)
        aligned = d.reindex(full_grid)

        mentions = aligned[MENTIONS_COL]
        kept_mask = mentions.ge(MIN_MENTIONS_MONTH)

        months_kept = int(kept_mask.sum())
        months_dropped_low_mentions = int((mentions.notna() & ~kept_mask).sum())

        # “missing” for gap calc = months not kept (either absent or below threshold)
        missing_mask = mentions.isna() | ~kept_mask
        missing_idx = aligned.index[missing_mask]
        max_gap = longest_consecutive_gap(missing_idx, full_grid)

        kept_idx = aligned.index[kept_mask]
        first_kept = kept_idx.min() if months_kept > 0 else pd.NaT
        last_kept = kept_idx.max() if months_kept > 0 else pd.NaT

        coverage_ratio = months_kept / months_full if months_full > 0 else np.nan

        kept_mentions = mentions[kept_mask]
        avg_mentions_kept = float(kept_mentions.mean()) if months_kept > 0 else np.nan
        median_mentions_kept = float(kept_mentions.median()) if months_kept > 0 else np.nan

        gate_pass = (months_kept >= MIN_MONTHS_AFTER_FILTER) and (max_gap <= MAX_CONSEC_GAP)

        audit_rows.append({
            "layer": layer_name,
            "group": g,
            "aspect": a,
            "period_full": f"{YEARS_START}..{YEARS_END}",
            "months_full": months_full,
            "min_mentions_threshold": MIN_MENTIONS_MONTH,
            "months_kept": months_kept,
            "months_dropped_low_mentions": months_dropped_low_mentions,
            "coverage_ratio_full": round(coverage_ratio, 4),
            "first_kept": None if pd.isna(first_kept) else first_kept.strftime("%Y-%m"),
            "last_kept": None if pd.isna(last_kept) else last_kept.strftime("%Y-%m"),
            "max_consec_gap": int(max_gap),
            "avg_mentions_kept": round(avg_mentions_kept, 2) if not np.isnan(avg_mentions_kept) else np.nan,
            "median_mentions_kept": round(median_mentions_kept, 2) if not np.isnan(median_mentions_kept) else np.nan,
            "gate_pass": bool(gate_pass),
            "note": "" if gate_pass else "fails_gate"
        })

    audit = pd.DataFrame(audit_rows).sort_values(["layer","group","aspect"])
    return audit

def summarize(audit_df):
    total = len(audit_df)
    passed = int(audit_df["gate_pass"].sum())
    by_aspect = (audit_df
                 .groupby("aspect")["gate_pass"]
                 .agg(series="count", passed="sum")
                 .assign(pass_rate=lambda x: (x["passed"] / x["series"]).round(3)))
    return total, passed, by_aspect

def main():
    ap = argparse.ArgumentParser(description="Preprocessing audit for GLOBAL/BRAND layers.")
    ap.add_argument("--asi_csv", required=True, help="Path to ASI CSV (global or brand layer).")
    ap.add_argument("--layer", required=True, choices=["GLOBAL","BRAND"], help="Which layer this file represents.")
    ap.add_argument("--out_csv", required=True, help="Where to write the audit CSV.")
    args = ap.parse_args()

    inp = Path(args.asi_csv)
    outp = Path(args.out_csv)
    outp.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(inp)
    audit = audit_one_layer(df, args.layer)
    audit.to_csv(outp, index=False)

    total, passed, by_aspect = summarize(audit)
    print(f"[{args.layer}] wrote audit: {outp}")
    print(f"  series total = {total}, passed gate = {passed} ({passed/total:.1%})")
    print("  by aspect (series / passed / pass_rate):")
    print(by_aspect.to_string())

if __name__ == "__main__":
    main()
