#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cpd_simple.py
Parametric change-point and trend analysis for ABSA monthly results.

Examples:
  python cpd_simple.py \
    --asi_csv "outputs/asi/asi_monthly.brand.csv" \
    --group "HP" \
    --aspect "price"

  python cpd_simple.py \
    --asi_csv "outputs/asi/asi_monthly.global.csv" \
    --group "GLOBAL" \
    --aspect "performance"
"""
import argparse
import os
import sys
import math
import json
from typing import List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



try:
    import ruptures as rpt
except Exception:
    rpt = None


import re
from pathlib import Path

def _safe_name(s: str) -> str:
    """Make strings safe for Windows/macOS/Linux filesystems."""
    s = str(s)
    # Replace path separators and Windows-illegal chars with underscores
    s = re.sub(r'[<>:"/\\|?*\n\r\t]+', '_', s)
    # Trim trailing spaces/dots which Windows dislikes on filenames
    return s.strip(' .')


def _fail(msg: str, code: int = 2):
    print(f"[ERROR] {msg}", file=sys.stderr)
    sys.exit(code)

def parse_args():
    p = argparse.ArgumentParser(
        description="Detect temporal/sudden sentiment shifts at aspect level (ABSA).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Data params
    p.add_argument("--asi_csv", required=True, help="Path to ASI CSV (monthly ABSA output).")
    p.add_argument("--group", required=True, help="Group key to filter (brand name, GLOBAL, family, or product).")
    p.add_argument("--aspect", required=True, help="Aspect to filter (e.g., price, performance).")
    p.add_argument("--group_col", default="group", help="Column name for group.")
    p.add_argument("--aspect_col", default="aspect", help="Column name for aspect.")
    p.add_argument("--date_col", default="month", help="Column name for timestamp.")
    p.add_argument("--value_col", default="ASI", choices=["ASI", "p_pos", "p_neg", "p_neu"],
                   help="Which metric to analyze.")
    p.add_argument("--mentions_col", default="mentions", help="Column with total mentions.")
    p.add_argument("--min_mentions", type=int, default=0, help="Filter out months with < min_mentions.")
    p.add_argument("--case_insensitive", action="store_true", help="Case-insensitive matching for group/aspect.")
    # Smoothing/normalization
    p.add_argument("--smoothing_window", type=int, default=3, help="Rolling mean window (0 or 1 to disable).")
    p.add_argument("--std_window", type=int, default=6, help="Rolling std window for volatility.")
    p.add_argument("--normalize", action="store_true", help="Z-score normalize the (optionally smoothed) series.")
    # Change-point
    p.add_argument("--method", default="pelt", choices=["pelt", "binseg", "bottomup", "window"],
                   help="Ruptures algorithm.")
    p.add_argument("--model", default="rbf", choices=["l1", "l2", "rbf", "linear", "normal", "rank"],
                   help="Cost model for ruptures.")
    p.add_argument("--n_bkps", type=int, default=None, help="Number of change points (alternative to --pen).")
    p.add_argument("--pen", type=str, default=None,
                   help="Penalty. Float (e.g., 6.0) or one of: aic, bic, hq, sqrt, log.")
    p.add_argument("--min_size", type=int, default=2, help="Minimum segment length for ruptures.")
    p.add_argument("--jump", type=int, default=1, help="Sub-sampling in ruptures (1 = full).")
    # Trend
    p.add_argument("--slope_window", type=int, default=6, help="Window size to estimate rolling slope (months).")
    # Output
    p.add_argument("--output_dir", default="outputs/cpd", help="Where to write figures and CSVs.")
    p.add_argument("--prefix", default=None, help="Optional filename prefix.")
    p.add_argument("--no_plot", action="store_true", help="Disable matplotlib plot export.")
    p.add_argument("--save_json", action="store_true", help="Also save a JSON summary of detections.")
    return p.parse_args()

def read_and_filter(args) -> pd.DataFrame:
    if not os.path.exists(args.asi_csv):
        _fail(f"CSV not found: {args.asi_csv}")
    df = pd.read_csv(args.asi_csv)
    for col in [args.group_col, args.aspect_col, args.date_col, args.value_col]:
        if col not in df.columns:
            _fail(f"Column '{col}' missing in CSV. Found columns: {list(df.columns)}")

    if args.case_insensitive:
        m_group = df[args.group_col].astype(str).str.lower() == args.group.lower()
        m_aspect = df[args.aspect_col].astype(str).str.lower() == args.aspect.lower()
    else:
        m_group = df[args.group_col].astype(str) == args.group
        m_aspect = df[args.aspect_col].astype(str) == args.aspect

    d = df[m_group & m_aspect].copy()
    if d.empty:
        _fail(f"No rows for group='{args.group}' & aspect='{args.aspect}'.")

    d[args.date_col] = pd.to_datetime(d[args.date_col])
    d = d.sort_values(args.date_col).drop_duplicates(subset=[args.date_col], keep="last").reset_index(drop=True)

    if args.min_mentions and args.mentions_col in d.columns:
        d = d[d[args.mentions_col] >= args.min_mentions].copy()

    return d

def rolling_slope(y: np.ndarray, window: int) -> np.ndarray:
    if window <= 1: return np.full_like(y, np.nan, dtype=float)
    n = len(y)
    out = np.full(n, np.nan, dtype=float)
    x = np.arange(window) - (window - 1) / 2
    denom = np.sum(x**2)
    for i in range(window-1, n):
        seg = y[i-window+1:i+1]
        y_c = seg - seg.mean()
        out[i] = np.dot(x, y_c) / denom
    return out

def choose_algo(args):
    if rpt is None:
        _fail("ruptures is not installed. Install with: pip install ruptures")
    common = dict(model=args.model, min_size=args.min_size, jump=args.jump)
    if args.method == "pelt": return rpt.Pelt(**common)
    if args.method == "binseg": return rpt.Binseg(**common)
    if args.method == "bottomup": return rpt.BottomUp(**common)
    if args.method == "window": return rpt.Window(width=max(2, args.min_size*2), model=args.model, jump=args.jump)
    _fail(f"Unknown method: {args.method}")

def parse_pen(pen_str: Optional[str], n: int) -> Optional[float]:
    if pen_str is None: return None
    try: return float(pen_str)
    except ValueError: pass
    if pen_str.lower() == "sqrt": return math.sqrt(n)
    if pen_str.lower() == "log": return math.log(max(n, 2))
    if pen_str.lower() == "aic": return 2.0 * math.log(max(n, 2))
    if pen_str.lower() == "bic": return math.log(max(n, 2)) * math.log(max(n, 2))
    if pen_str.lower() in ["hq","hqic"]: return 2.0 * math.log(math.log(max(n, 3)))
    _fail(f"Unrecognized --pen value: {pen_str}")

def detect_changepoints(series: np.ndarray, args) -> List[int]:
    algo = choose_algo(args).fit(series)
    n = len(series)
    if args.pen is not None:
        return algo.predict(pen=parse_pen(args.pen, n))
    if args.n_bkps is not None:
        return algo.predict(n_bkps=args.n_bkps)
    return algo.predict(pen=parse_pen("bic", n))

def annotate_segments(df, idxs, value_col):
    n = len(df)
    ends = sorted([i for i in idxs if 0 < i <= n])
    if len(ends) == 0 or ends[-1] != n: ends.append(n)
    seg_id = np.zeros(n, dtype=int)
    seg_means = []
    start, sid = 0, 0
    for end in ends:
        seg_id[start:end] = sid
        seg_means.append(df[value_col].iloc[start:end].mean())
        start, sid = end, sid+1
    df = df.copy()
    df["segment_id"] = seg_id
    df["segment_mean"] = [seg_means[s] for s in seg_id]
    cp_flags = np.zeros(n, dtype=bool)
    for e in ends[:-1]:
        if 0 <= e-1 < n: cp_flags[e-1] = True
    df["is_change_point"] = cp_flags
    df["delta_to_next_seg"] = np.nan
    for i, e in enumerate(ends[:-1]):
        row = e-1
        if 0 <= row < n:
            df.loc[df.index[row], "delta_to_next_seg"] = seg_means[i+1] - seg_means[i]
    return df

def make_plot(df, args, title, png_path):
    fig, ax = plt.subplots(figsize=(11, 4.5))
    t = df[args.date_col]
    ax.plot(t, df["_series"], label=args.value_col, linewidth=1.8)
    if "_series_smooth" in df.columns:
        ax.plot(t, df["_series_smooth"], label=f"smoothed (w={args.smoothing_window})", linewidth=1.6, alpha=0.9)
    for i in df.index[df["is_change_point"]]:
        ax.axvline(t.iloc[i], linestyle="--", alpha=0.7)
    ax.set_title(title)
    ax.set_xlabel(args.date_col)
    ax.set_ylabel(args.value_col + (" (z)" if args.normalize else ""))
    ax.legend(loc="best"); ax.grid(True, alpha=0.25)
    fig.tight_layout(); fig.savefig(png_path, dpi=150); plt.close(fig)

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # Build a safe prefix for files (no unintended subfolders)
    group_safe = _safe_name(args.group)
    aspect_safe = _safe_name(args.aspect)
    val_safe = _safe_name(args.value_col)

    if args.prefix:
        prefix = _safe_name(args.prefix)
    else:
        prefix = f"{group_safe}__{aspect_safe}__{val_safe}"

    # Use pathlib for clarity and ensure parent exists
    base_path = Path(args.output_dir) / prefix
    base_path.parent.mkdir(parents=True, exist_ok=True)
    base = str(base_path)

    d = read_and_filter(args)
    if len(d) < max(5, args.min_size*2+1):
        _fail(f"Not enough data points: {len(d)}")

    series = d[args.value_col].astype(float).to_numpy()
    d["_series"] = series.copy()

    if args.smoothing_window > 1:
        d["_series_smooth"] = pd.Series(series).rolling(args.smoothing_window, min_periods=1, center=True).mean().values
        series_used = d["_series_smooth"].to_numpy()
    else:
        series_used = series.copy()

    if args.normalize:
        mu, sd = np.nanmean(series_used), np.nanstd(series_used) or 1.0
        series_used = (series_used - mu)/sd
        d["_series"] = (d["_series"] - mu)/sd
        if "_series_smooth" in d.columns:
            d["_series_smooth"] = (d["_series_smooth"] - mu)/sd

    d["rolling_std"] = pd.Series(series_used).rolling(args.std_window, min_periods=2).std().values
    d["rolling_slope"] = rolling_slope(series_used, args.slope_window)

    cp_idxs = detect_changepoints(series_used, args)
    d = annotate_segments(d, cp_idxs, "_series")

    cp_rows = d[d["is_change_point"]]
    cp_table = pd.DataFrame({
        "cp_date": cp_rows[args.date_col].dt.strftime("%Y-%m-%d"),
        "delta_to_next_seg": cp_rows["delta_to_next_seg"].round(4),
        "prev_seg_mean": cp_rows["segment_mean"].round(4)
    })
    if not cp_table.empty:
        next_means = []
        for i in cp_rows.index:
            sid = int(d.loc[i, "segment_id"])
            nxt = d[d["segment_id"] == sid+1]["segment_mean"].iloc[0] if (sid+1) in d["segment_id"].values else np.nan
            next_means.append(round(float(nxt),4) if not np.isnan(nxt) else np.nan)
        cp_table["next_seg_mean"] = next_means

    d.to_csv(base+".timeseries.csv", index=False)
    cp_table.to_csv(base+".changepoints.csv", index=False)
    if not args.no_plot:
        make_plot(d, args, f"{args.group} · {args.aspect} · {args.value_col}", base+".png")
    if args.save_json:
        with open(base+".summary.json","w",encoding="utf-8") as f:
            json.dump({"params":vars(args),"changepoints":cp_table.to_dict(orient="records")}, f, indent=2)

    print(f"Detected {len(cp_table)} change points. ")

if __name__ == "__main__":
    main()
