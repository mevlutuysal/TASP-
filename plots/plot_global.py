#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


# ---------- helpers ----------
def canonical(s: str) -> str:
    s = (s or "").strip()
    return "".join(ch for ch in s.lower().replace(" ", "_") if ch.isalnum() or ch == "_")

def parse_month_col(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce", dayfirst=True).dt.to_period("M").dt.to_timestamp()

def pick_month_col(df: pd.DataFrame) -> str | None:
    # 1) exact 'month'
    for c in df.columns:
        if c.lower() == "month":
            return c
    # 2) contains 'month' but not year-ish tokens
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
        raise FileNotFoundError(f"[plot_global] not found: layer={layer} group={group} aspect='{aspect}' under {root}")

    # try both cp filename conventions
    cp1 = match.with_name(match.name.replace(".timeseries.csv", ".changepoints.csv"))
    cp2 = match.with_name(match.name.replace(".timeseries", ".changepoints"))
    cp_path = cp1 if cp1.exists() else (cp2 if cp2.exists() else None)
    return match, cp_path

def load_timeseries(ts_path: Path, date_col_override: str | None) -> pd.DataFrame:
    df = pd.read_csv(ts_path)
    dcol = date_col_override or pick_month_col(df)
    ycol = pick_value_col(df)
    if dcol is None:
        raise ValueError(f"[plot_global] no monthly date column in {ts_path}. Use --date_col MONTH_COL")
    if ycol is None:
        raise ValueError(f"[plot_global] no numeric value column in {ts_path}")

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
    months = parse_month_col(cps[ccol]).dropna().sort_values().tolist()
    return months


# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cpd_root", required=True)
    ap.add_argument("--aspect", required=True)
    ap.add_argument("--cp_month", help="YYYY-MM to highlight")
    ap.add_argument("--no_highlight", action="store_true")
    ap.add_argument("--date_col", help="force the monthly date column (e.g., 'month')")
    ap.add_argument("--cp_col", help="force the change-point date column (e.g., 'cp_month')")
    ap.add_argument("--pen", type=float, default=4.0)
    ap.add_argument("--min_size", type=int, default=4)
    ap.add_argument("--smoothing", type=int, default=1)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    ts_path, cp_path = locate_series(args.cpd_root, "GLOBAL", "GLOBAL",
                                     args.aspect, args.pen, args.min_size, args.smoothing, "ASI")
    ts = load_timeseries(ts_path, args.date_col)
    cp_months = load_cp_months(cp_path, args.cp_col)

    fig, ax = plt.subplots(figsize=(10, 4.2))
    ax.plot(ts["month"], ts["y"], linewidth=1.8)
    for m in cp_months:
        ax.axvline(m, linestyle="--", linewidth=1, alpha=0.6)
    if args.cp_month and not args.no_highlight:
        ax.axvline(pd.to_datetime(args.cp_month + "-01"), linewidth=2.2)

    ax.set_title(f"GLOBAL · {args.aspect} (z-normalized ASI)")
    ax.set_xlabel("Month")
    ax.set_ylabel("ASI (z)")
    ax.grid(alpha=0.25)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(args.out, dpi=160)
    print(f"Saved {args.out}")

if __name__ == "__main__":
    main()
