#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Run your existing cpd.py over a small parameter grid and across many (group, aspect) series.

- BRAND layer uses an audit CSV to restrict to passed series.
- GLOBAL layer runs for all aspects of group == "GLOBAL" (or all groups if you want; see flag).

Outputs go under:
  <out_root>/<LAYER>/pen{P}_ms{M}_sw{S}/val_<VALUE_COL>/
…so each setting has its own folder and files from cpd.py won't overwrite each other.

Example PowerShell:
  python run_cpd_grid.py `
    --layer BRAND `
    --asi_csv "outputs/final/asi_monthly.brand.csv" `
    --audit_csv "outputs/audit/brand_audit.csv" `
    --out_root "outputs/cpd_grid" `
    --pens 2,3,4,5 --min_sizes 2,4,6 --smoothings 1 `
    --include_ppos --include_pneg

  python run_cpd_grid.py `
    --layer GLOBAL `
    --asi_csv "outputs/final/asi_monthly.global.csv" `
    --out_root "outputs/cpd_grid" `
    --pens 2,3,4,5 --min_sizes 2,4,6 --smoothings 1 `
    --include_ppos --include_pneg
"""

import argparse, itertools, os, subprocess, sys, re
from pathlib import Path
import pandas as pd

CANONICAL_ASPECTS = [
    "Performance","Operating Systems","Input Devices","Display","Battery",
    "Design/Build","Audio","Camera","Connectivity","Price","Support"
]

def parse_csv_list(s):
    return [x.strip() for x in s.split(",") if x.strip()]

def load_series_brand(asi_csv, audit_csv, aspects=None):
    """Return list of (group, aspect) that passed the gate in BRAND audit."""
    aud = pd.read_csv(audit_csv)
    aud = aud[aud["gate_pass"] == True]
    if aspects:
        aud = aud[aud["aspect"].isin(aspects)]
    # Keep only the five brands you said you focus on if present
    focus = {"Acer","Asus","ASUS","Dell","HP","Lenovo"}
    if "group" in aud.columns:
        aud = aud[aud["group"].astype(str).isin(focus)] if any(aud["group"].astype(str).isin(focus)) else aud
    pairs = sorted({(str(r["group"]), str(r["aspect"])) for _, r in aud.iterrows()})
    return pairs

def load_series_global(asi_csv, aspects=None, group_value="GLOBAL"):
    """Return list of (group, aspect) for GLOBAL layer."""
    df = pd.read_csv(asi_csv)
    # tolerate case differences
    if "group" not in df.columns or "aspect" not in df.columns:
        raise ValueError("CSV must have 'group' and 'aspect' columns.")
    # Prefer group==GLOBAL; if not found, take unique groups anyway.
    if group_value in df["group"].astype(str).unique():
        df = df[df["group"].astype(str) == group_value]
    if aspects:
        df = df[df["aspect"].astype(str).isin(aspects)]
    pairs = sorted({(str(g), str(a)) for g, a in df[["group","aspect"]].itertuples(index=False, name=None)})
    return pairs

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", required=True, choices=["BRAND","GLOBAL"])
    ap.add_argument("--asi_csv", required=True)
    ap.add_argument("--audit_csv", help="Required for BRAND layer.")
    ap.add_argument("--out_root", required=True)
    ap.add_argument("--pens", default="2,3,4,5")
    ap.add_argument("--min_sizes", default="2,4,6")
    ap.add_argument("--smoothings", default="1")  # "1,3" for extra
    ap.add_argument("--aspects", default="")      # optional subset
    ap.add_argument("--method", default="pelt")
    ap.add_argument("--model", default="l2")
    ap.add_argument("--std_window", type=int, default=6)
    ap.add_argument("--slope_window", type=int, default=6)
    ap.add_argument("--include_ppos", action="store_true")
    ap.add_argument("--include_pneg", action="store_true")
    ap.add_argument("--cpd_py", default="scripts/cpd.py", help="Path to your cpd.py")
    ap.add_argument("--global_group_value", default="GLOBAL", help="Value used in 'group' column for global series")
    args = ap.parse_args()

    # pens = [float(x) for x in parse_csv_list(args.pens)]
    pens = []
    for p in args.pens.split(","):
        p = p.strip()
        if p.lower() == "bic":
            pens.append("bic")  # keep as string
        else:
            pens.append(float(p))  # numeric values

    min_sizes = [int(x) for x in parse_csv_list(args.min_sizes)]
    smoothings = [int(x) for x in parse_csv_list(args.smoothings)]
    aspects = parse_csv_list(args.aspects) if args.aspects else None

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    # Collect series
    if args.layer == "BRAND":
        if not args.audit_csv:
            print("ERROR: --audit_csv is required for BRAND layer", file=sys.stderr)
            sys.exit(2)
        series = load_series_brand(args.asi_csv, args.audit_csv, aspects=aspects)
    else:
        series = load_series_global(args.asi_csv, aspects=aspects, group_value=args.global_group_value)

    if not series:
        print("No (group, aspect) series found. Check inputs.", file=sys.stderr)
        sys.exit(1)

    # Values to analyze
    value_cols = ["ASI"]
    if args.include_ppos: value_cols.append("p_pos")
    if args.include_pneg: value_cols.append("p_neg")

    total_jobs = len(series) * len(value_cols) * len(pens) * len(min_sizes) * len(smoothings)
    print(f"[{args.layer}] Running grid: series={len(series)}, values={value_cols}, "
          f"pens={pens}, min_sizes={min_sizes}, smoothings={smoothings} -> jobs={total_jobs}")

    # Run grid
    job_idx = 0
    for (group, aspect) in series:
        for val in value_cols:
            for P, M, S in itertools.product(pens, min_sizes, smoothings):
                job_idx += 1
                # Directory that encodes settings
                # Directory that encodes settings
                p_tag = f"{P:g}" if isinstance(P, (int, float)) else str(P)
                out_dir = out_root / args.layer / f"pen{p_tag}_ms{M}_sw{S}" / f"val_{val}"
                out_dir.mkdir(parents=True, exist_ok=True)

                cmd = [
                    sys.executable, args.cpd_py,
                    "--asi_csv", args.asi_csv,
                    "--group", group,
                    "--aspect", aspect,
                    "--method", args.method,
                    "--model", args.model,
                    "--smoothing_window", str(S),
                    "--normalize",           # always normalize for main runs
                    "--pen", str(P),
                    "--min_size", str(M),
                    "--std_window", str(args.std_window),
                    "--slope_window", str(args.slope_window),
                    "--output_dir", str(out_dir),
                ]
                if val != "ASI":
                    cmd.extend(["--value_col", val])

                # Launch
                # print(f"[{job_idx}/{total_jobs}] {group} · {aspect} · {val} | pen={P} ms={M} sw={S}")
                proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
                # Optional: write a log per job

                def _safe_name(s: str) -> str:
                    # Replace Windows-illegal characters and slashes with underscores
                    # Illegal on Windows: <>:"/\|?*  and also strip trailing dots/spaces
                    s = re.sub(r'[<>:"/\\|?*\n\r\t]+', '_', str(s))
                    s = s.strip(' .')
                    return s

                # ...

                # Optional: keep the print exactly as-is (nice for console)
                print(f"[{job_idx}/{total_jobs}] {group} · {aspect} · {val} | pen={P} ms={M} sw={S} dcp={proc.stdout}")

                # Safe log filename
                log_name = f"run_{_safe_name(group)}__{_safe_name(aspect)}__{_safe_name(val)}.log"
                with open(out_dir / log_name, "w", encoding="utf-8") as f:
                    f.write(proc.stdout)

    print("Grid run complete.")

if __name__ == "__main__":
    main()
