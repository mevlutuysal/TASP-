import argparse, json
from datetime import datetime, timezone

DATE_KEYS = ["date","review_date","review_time","timestamp","time"]

def parse_date_any(obj):
    for k in DATE_KEYS:
        if k in obj and obj[k] is not None:
            v = obj[k]
            # numeric epoch?
            if isinstance(v,(int,float)):
                val=int(v)
                try:
                    if val>1_000_000_000_000: dt=datetime.fromtimestamp(val/1000, tz=timezone.utc)
                    elif val>10_000_000_000:   dt=datetime.fromtimestamp(val/1000, tz=timezone.utc)
                    else:                       dt=datetime.fromtimestamp(val, tz=timezone.utc)
                    return dt.date()
                except: pass
            # strings
            s=str(v).strip()
            if not s: continue
            try:
                return datetime.fromisoformat(s.replace("Z","+00:00")).date()
            except: pass
            for fmt in ("%Y-%m-%d","%Y/%m/%d","%d-%m-%Y","%m/%d/%Y"):
                try:    return datetime.strptime(s, fmt).date()
                except: pass
    return None

def main():
    ap=argparse.ArgumentParser(description="Filter JSONL by inclusive date range.")
    ap.add_argument("--in_jsonl", required=True)
    ap.add_argument("--out_jsonl", required=True)
    ap.add_argument("--start", required=True, help="YYYY-MM-DD")
    ap.add_argument("--end",   required=True, help="YYYY-MM-DD (inclusive)")
    ap.add_argument("--drop_if_no_date", action="store_true", help="Skip rows with unparseable date")
    args=ap.parse_args()

    start = datetime.strptime(args.start, "%Y-%m-%d").date()
    end   = datetime.strptime(args.end,   "%Y-%m-%d").date()

    n=kept=nodate=0
    with open(args.in_jsonl,"r",encoding="utf-8",errors="ignore") as fin, \
         open(args.out_jsonl,"w",encoding="utf-8") as fout:
        for line in fin:
            line=line.strip()
            if not line: continue
            try: obj=json.loads(line)
            except: continue
            n+=1
            d=parse_date_any(obj)
            if d is None:
                if args.drop_if_no_date: nodate+=1; continue
                else: continue  # safer for temporal work: skip if date unknown
            if start <= d <= end:
                fout.write(json.dumps(obj, ensure_ascii=False)+"\n")
                kept+=1
    print(f"[OK] read={n:,} kept={kept:,} skipped_no_date={nodate:,} wrote={args.out_jsonl}")

if __name__=="__main__":
    main()
