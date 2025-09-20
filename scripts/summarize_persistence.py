#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Sweep all *.changepoints.csv produced by run_cpd_grid.py and compute:
- persistence of CP months across grid settings (±1 month tolerance),
- median Δ mean (z) across settings that contain that CP,
- short-window standardized effect around the CP for one baseline setting (pen=4, min_size=4, sw=1 if available).

Outputs a tidy CSV you can sort by persistence × |Δ|.
"""

import argparse, glob, os, sys, re
from pathlib import Path
import pandas as pd
import numpy as np
from collections import defaultdict

def month_floor(ts):
    dt = pd.to_datetime(ts, errors="coerce")
    # floor to month start without specifying 'MS'
    return dt.dt.to_period("M").dt.to_timestamp()


def read_cp_csv(path):
    try:
        df = pd.read_csv(path)
        if df.empty:
            return df

        # 1) Best case: cp_month already present
        if "cp_month" in df.columns:
            df["cp_month"] = month_floor(df["cp_month"])

        # 2) Else derive cp_month from cp_date
        elif "cp_date" in df.columns:
            df["cp_month"] = month_floor(df["cp_date"])

        # 3) Else try common variants (any col containing "date" OR obvious names)
        else:
            cand_cols = [c for c in df.columns
                         if ("date" in c.lower()) or (c.lower() in {"month", "timestamp"})]
            if not cand_cols:
                return pd.DataFrame()  # nothing to normalize → skip this file
            df["cp_month"] = month_floor(df[cand_cols[0]])

        # Keep core columns if present
        keep = [c for c in ["cp_month", "delta_to_next_seg", "prev_seg_mean", "next_seg_mean"]
                if c in df.columns]
        # Always keep cp_month
        if "cp_month" not in keep:
            keep = ["cp_month"] + keep

        return df[keep].dropna(subset=["cp_month"])
    except Exception:
        return pd.DataFrame()


def read_timeseries_csv(path, date_col_guess=("month","date","timestamp")):
    try:
        df = pd.read_csv(path)
        # find date col
        dc = None
        for c in date_col_guess:
            if c in df.columns:
                dc = c; break
        if not dc:
            # fallback: first col that looks like date
            for c in df.columns:
                if "month" in c.lower() or "date" in c.lower():
                    dc = c; break
        if not dc: return None, None
        df[dc] = month_floor(df[dc])
        # choose value col
        val = None
        for c in ["_series","ASI","p_pos","p_neg","p_neu"]:
            if c in df.columns:
                val = c; break
        if not val: return None, None
        return df[[dc,val]].rename(columns={dc:"month", val:"y"}), "y"
    except Exception:
        return None, None

def cohen_d(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    if len(a) < 2 or len(b) < 2: return np.nan
    ma, mb = np.nanmean(a), np.nanmean(b)
    sa, sb = np.nanstd(a, ddof=1), np.nanstd(b, ddof=1)
    # pooled sd
    n1, n2 = len(a), len(b)
    sp = np.sqrt(((n1-1)*sa**2 + (n2-1)*sb**2) / max(n1+n2-2,1))
    if sp == 0 or np.isnan(sp): return np.nan
    return (mb - ma) / sp

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_root", required=True, help="Root from run_cpd_grid.py, e.g., outputs/cpd_grid/BRAND")
    ap.add_argument("--layer", required=True, choices=["BRAND","GLOBAL"])
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--tolerance_months", type=int, default=1, help="± months for considering same CP across runs")
    ap.add_argument("--baseline_pen", default="4.0", help="Baseline penalty (float or 'bic')")
    ap.add_argument("--baseline_min_size", type=int, default=4)
    ap.add_argument("--baseline_sw", type=int, default=1)
    args = ap.parse_args()

    in_root = Path(args.in_root)
    # Gather all changepoint CSVs
    cp_paths = list(in_root.glob("pen*_ms*_sw*/val_*/**/*.changepoints.csv"))
    print(f"Found {len(cp_paths)} changepoint CSVs under {in_root}")

    if not cp_paths:
        print("No changepoints found. Did you run the grid?", file=sys.stderr)
        return

    # Parse paths to extract setting + series identifiers
    records = []
    for p in cp_paths:
        parts = p.parts

        # find the settings folder like "pen2_ms2_sw1" (or "penbic_ms4_sw1")
        # setting = next((x for x in parts if x.startswith("pen") and "_ms" in x and "_sw" in x), None)
        # val_dir = next((x for x in parts if x.startswith("val_")), None)
        # if setting is None or val_dir is None:
        #     continue

        # find the single part that encodes all three: pen*_ms*_sw*
        try:
            settings_part = [x for x in p.parts if x.startswith("pen") and "_ms" in x and "_sw" in x][0]
            # supports pen as float (e.g., 4, 4.0) or string (e.g., bic)
            # examples: "pen4_ms2_sw1", "pen3.5_ms4_sw1", "penbic_ms4_sw1"
            m = re.match(r"pen(?P<pen>[^_]+)_ms(?P<ms>\d+)_sw(?P<sw>\d+)", settings_part, re.IGNORECASE)
            if not m:
                continue
            pen_raw = m.group("pen")
            try:
                P = float(pen_raw)
            except ValueError:
                P = pen_raw.lower()  # keep string like "bic"

            M = int(m.group("ms"))
            S = int(m.group("sw"))

            val_dir = [x for x in p.parts if x.startswith("val_")][0]
            value_col = val_dir.replace("val_", "")
        except (IndexError, ValueError):
            continue

        # series id from filename prefix: GROUP__ASPECT__VALUE
        stem = p.stem  # e.g., ASUS__performance__ASI.changepoints
        name = stem.replace(".changepoints", "")
        if "__" in name:
            parts2 = name.split("__")
            group, aspect = parts2[0], parts2[1]
        else:
            group, aspect = "UNKNOWN", "UNKNOWN"

        df = read_cp_csv(p)
        if df is None or df.empty:
            continue
        for _, r in df.iterrows():
            records.append({
                "group": group,
                "aspect": aspect,
                "value_col": value_col,
                "pen": str(P), "min_size": M, "sw": S,
                "cp_month": r["cp_month"],
                "delta": r.get("delta_to_next_seg", np.nan)
            })

    if not records:
        print("No CP rows found in grid outputs.", file=sys.stderr); return

    cpdf = pd.DataFrame(records)
    # Compute persistence: for each series, cluster by month with ±tolerance
    # We'll do it by rounding months and counting presence in a sliding window.

    def month_bins(series_months, tol):
        # returns dict: month -> count where any cp within ±tol falls into that key month
        if not len(series_months): return {}
        unique_months = pd.to_datetime(sorted(series_months.unique()))
        bins = {}
        for m in unique_months:
            # count how many settings have a cp within ±tol of m
            close = []
            for mm in series_months:
                if abs((pd.to_datetime(mm) - m).days) <= 31*tol:
                    close.append(mm)
            bins[m] = len(close)
        return bins

    out_rows = []
    for (g, a, v), df1 in cpdf.groupby(["group","aspect","value_col"]):
        # number of unique settings for this series
        n_settings = len(df1[["pen","min_size","sw"]].drop_duplicates())
        # candidate months to consider (unique over all runs)
        candidates = pd.to_datetime(sorted(df1["cp_month"].unique()))
        # For each candidate, count presence across settings (within ± tol)
        for cand in candidates:
            # a setting is counted present if that setting has ANY cp within ± tol of cand
            present = (
                df1.assign(diff_days=(df1["cp_month"] - cand).dt.days.abs())
                   .query("diff_days <= @args.tolerance_months*31")
                   .groupby(["pen","min_size","sw"], as_index=False).size()
            )
            pers = 0.0 if n_settings == 0 else len(present) / n_settings

            # median delta across matching settings
            med_delta = (
                df1[df1["cp_month"].between(cand - pd.offsets.MonthBegin(args.tolerance_months),
                                            cand + pd.offsets.MonthEnd(args.tolerance_months))]
                ["delta"].median()
            )

            out_rows.append({
                "layer": args.layer,
                "group": g, "aspect": a, "value_col": v,
                "cp_month": cand.strftime("%Y-%m"),
                "n_settings": n_settings,
                "persistence": round(float(pers), 3),
                "median_delta_z": round(float(med_delta), 3) if pd.notna(med_delta) else np.nan
            })

    out = pd.DataFrame(out_rows)

    # Attach a short-window standardized effect using the baseline setting (if present)
    # We'll look for timeseries CSV in baseline folder and compute Cohen's d
    try:
        base_pen_str = f"{float(args.baseline_pen):g}"
    except ValueError:
        base_pen_str = args.baseline_pen.lower()

    base_dir = Path(args.in_root) / f"pen{base_pen_str}_ms{args.baseline_min_size}_sw{args.baseline_sw}"

    ts_paths = list(base_dir.glob("val_*/**/*.timeseries.csv"))
    ts_index = {}
    for p in ts_paths:
        # group__aspect__VALUE.timeseries.csv
        stem = p.stem.replace(".timeseries","")
        parts = stem.split("__")
        if len(parts) >= 3:
            g, a, v = parts[0], parts[1], parts[2]
            ts_index[(g,a,v)] = p

    def window_effect(ts_df, cp_month, w_pre=3, w_post=3):
        # compute Cohen's d for mean change over short windows
        if ts_df is None: return np.nan
        m = pd.to_datetime(f"{cp_month}-01")
        pre = ts_df[(ts_df["month"] >= (m - pd.DateOffset(months=w_pre))) & (ts_df["month"] < m)]["y"].values
        post= ts_df[(ts_df["month"] >= m) & (ts_df["month"] <= (m + pd.DateOffset(months=w_post-1)))]["y"].values
        # remove NaNs
        pre = pre[~np.isnan(pre)]; post = post[~np.isnan(post)]
        if len(pre) < 2 or len(post) < 2: return np.nan
        # Cohen's d
        ma, mb = np.mean(pre), np.mean(post)
        sa, sb = np.std(pre, ddof=1), np.std(post, ddof=1)
        sp = np.sqrt(((len(pre)-1)*sa**2 + (len(post)-1)*sb**2) / max(len(pre)+len(post)-2,1))
        if sp == 0 or np.isnan(sp): return np.nan
        return (mb - ma) / sp

    effects = []
    for i, r in out.iterrows():
        key = (r["group"], r["aspect"], r["value_col"])
        p = ts_index.get(key)
        if p:
            ts_df, _ = read_timeseries_csv(p)
            d = window_effect(ts_df, r["cp_month"], 3, 3)
        else:
            d = np.nan
        effects.append(np.nan if d is None else round(float(d), 3))
    out["short_window_cohen_d"] = effects

    # order columns nicely
    out = out[[
        "layer","group","aspect","value_col","cp_month","n_settings",
        "persistence","median_delta_z","short_window_cohen_d"
    ]].sort_values(["group","aspect","value_col","cp_month"])

    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out_csv, index=False)
    print(f"Wrote persistence summary: {args.out_csv}")
    print(out.head(12).to_string(index=False))

if __name__ == "__main__":
    main()
