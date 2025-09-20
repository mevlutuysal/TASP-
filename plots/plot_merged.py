#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


DEFAULT_BRANDS = ["ACER", "ASUS", "DELL", "HP", "LENOVO"]


# ---------- helpers ----------
def canonical(s: str) -> str:
    s = (s or "").strip()
    return "".join(ch for ch in s.lower().replace(" ", "_") if ch.isalnum() or ch == "_")

def parse_month_col(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce", dayfirst=True).dt.to_period("M").dt.to_timestamp()

def pick_month_col(df: pd.DataFrame) -> str | None:
    for c in df.columns:
        if c.lower() == "month":
            return c
    cands = [c for c in df.columns if "month" in c.lower()]
    cands = [c for c in cands if not any(tok in c.lower() for tok in ["year", "yr", "annual", "ann"])]
    return cands[0] if cands else None

def pick_value_col(df: pd.DataFrame) -> str | None:
    for c in ["ASI", "asi", "_series", "value", "y"]:
        if c in df.columns:
            return c
    for c in df.columns:
        if pd.api.types.is_numeric_dtype(df[c]):
            return c
    return None

def pick_cp_col(df: pd.DataFrame) -> str | None:
    for c in ["cp_month", "change_date", "change_month", "cp_date"]:
        if c in df.columns:
            return c
    cands = [c for c in df.columns if ("date" in c.lower() or "month" in c.lower())]
    cands = [c for c in cands if not any(tok in c.lower() for tok in ["year", "yr", "annual", "ann"])]
    return cands[0] if cands else None

def locate_series(cpd_root: str, layer: str, group: str, aspect: str,
                  pen: float, ms: int, sw: int, value_col: str = "ASI") -> tuple[Path, Path | None]:
    root = Path(cpd_root) / layer / f"pen{pen:g}_ms{ms}_sw{sw}" / f"val_{value_col}"
    key = canonical(aspect)
    match = None
    for p in root.glob(f"{group}__*__{value_col}.timeseries.csv"):
        mid = p.stem.split("__")[1] if "__" in p.stem else ""
        if canonical(mid) == key:
            match = p
            break
    if match is None:
        raise FileNotFoundError(f"[plot_merged] not found: layer={layer} group={group} aspect='{aspect}' under {root}")

    cp1 = match.with_name(match.name.replace(".timeseries.csv", ".changepoints.csv"))
    cp2 = match.with_name(match.name.replace(".timeseries", ".changepoints"))
    cp_path = cp1 if cp1.exists() else (cp2 if cp2.exists() else None)
    return match, cp_path

def load_timeseries(ts_path: Path, date_col_override: str | None) -> pd.DataFrame:
    df = pd.read_csv(ts_path)
    dcol = date_col_override or pick_month_col(df)
    ycol = pick_value_col(df)
    if dcol is None:
        raise ValueError(f"[plot_merged] no monthly date column in {ts_path}. Use --date_col MONTH_COL")
    if ycol is None:
        raise ValueError(f"[plot_merged] no numeric value column in {ts_path}")

    df["month"] = parse_month_col(df[dcol])
    df = df.dropna(subset=["month", ycol]).sort_values("month")
    df = df.groupby("month", as_index=False)[ycol].mean().rename(columns={ycol: "y"})
    return df

def load_cp_months(cp_path: Path | None, cp_col_override: str | None) -> list[pd.Timestamp]:
    if not cp_path or not cp_path.exists():
        return []
    cps = pd.read_csv(cp_path)
    ccol = cp_col_override or pick_cp_col(cps)
    if not ccol or ccol not in cps.columns:
        return []
    return parse_month_col(cps[ccol]).dropna().tolist()

def nearest_y(dfb: pd.DataFrame, month: pd.Timestamp) -> float:
    if (dfb["month"] == month).any():
        return float(dfb.loc[dfb["month"] == month, "y"].iloc[0])
    idx = (dfb["month"] - month).abs().argsort()
    return float(dfb.iloc[idx[0]]["y"])


# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cpd_root", required=True)
    ap.add_argument("--aspect", required=True)
    ap.add_argument("--global_cp", required=True, help="YYYY-MM anchor")
    ap.add_argument("--brands", default=",".join(DEFAULT_BRANDS))
    ap.add_argument("--layout", choices=["stacked", "single"], default="stacked")
    ap.add_argument("--no_highlight", action="store_true")
    ap.add_argument("--date_col", help="force the monthly date column (e.g., 'month')")
    ap.add_argument("--cp_col", help="force the change-point date column (e.g., 'cp_month')")
    ap.add_argument("--pen", type=float, default=4.0)
    ap.add_argument("--min_size", type=int, default=4)
    ap.add_argument("--smoothing", type=int, default=1)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    brands = [b.strip().upper() for b in args.brands.split(",") if b.strip()]

    # GLOBAL
    g_ts, g_cp = locate_series(args.cpd_root, "GLOBAL", "GLOBAL", args.aspect,
                               args.pen, args.min_size, args.smoothing, "ASI")
    g = load_timeseries(g_ts, args.date_col)
    g_cps = load_cp_months(g_cp, args.cp_col)
    m0 = pd.to_datetime(args.global_cp + "-01")

    # BRANDS
    aligned, brand_series = [], {}
    for b in brands:
        try:
            b_ts, b_cp = locate_series(args.cpd_root, "BRAND", b, args.aspect,
                                       args.pen, args.min_size, args.smoothing, "ASI")
        except FileNotFoundError:
            continue
        dfb = load_timeseries(b_ts, args.date_col)
        cps = load_cp_months(b_cp, args.cp_col)
        brand_series[b] = (dfb, cps)
        if any(abs((pd.Timestamp(m) - m0).days) <= 31 for m in cps):
            aligned.append(b)

    # Plot
    if args.layout == "stacked":
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True,
                                       gridspec_kw={"height_ratios": [2, 3]})
        # top/global
        ax1.plot(g["month"], g["y"], linewidth=1.8)
        for m in g_cps:
            ax1.axvline(m, linestyle="--", linewidth=1, alpha=0.6)
        if not args.no_highlight:
            ax1.axvline(m0, linewidth=2.2)
        ax1.set_title(f"GLOBAL · {args.aspect} (z-normalized ASI)")
        ax1.set_ylabel("GLOBAL ASI (z)")
        ax1.grid(alpha=0.25)

        # bottom/brands
        for b, (dfb, _) in brand_series.items():
            ax2.plot(dfb["month"], dfb["y"], linewidth=1.0, alpha=0.95, label=b)
        if not args.no_highlight:
            ax2.axvline(m0, linewidth=1.8)
        ax2.set_ylabel("Brand ASI (z)")
        ax2.set_xlabel("Month")
        ax2.grid(alpha=0.25)
        if aligned:
            ax2.legend(ncol=5, fontsize=9, frameon=False, title=f"Aligned: {', '.join(aligned)}")
        else:
            ax2.legend(ncol=5, fontsize=9, frameon=False)

    else:
        # single panel overlay
        fig, ax = plt.subplots(figsize=(10, 5.2))
        ax.plot(g["month"], g["y"], linewidth=2.0, label="GLOBAL")
        for m in g_cps:
            ax.axvline(m, linestyle="--", linewidth=1, alpha=0.6)
        for b, (dfb, _) in brand_series.items():
            ax.plot(dfb["month"], dfb["y"], linewidth=1.0, alpha=0.9, label=b)
        if not args.no_highlight:
            ax.axvline(m0, linewidth=2.0)
        # mark aligned brands
        for b, (dfb, cps) in brand_series.items():
            if any(abs((pd.Timestamp(m) - m0).days) <= 31 for m in cps):
                y = nearest_y(dfb, m0)
                ax.scatter([m0], [y], s=30)
        ax.set_title(f"{args.aspect} · GLOBAL + brands (z-normalized ASI)")
        ax.set_xlabel("Month")
        ax.set_ylabel("ASI (z)")
        ax.grid(alpha=0.25)
        ax.legend(ncol=6, frameon=False)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(args.out, dpi=160)
    print(f"Saved {args.out}")

if __name__ == "__main__":
    main()
