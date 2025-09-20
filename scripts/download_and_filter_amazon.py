
import argparse
import json
from pathlib import Path
from typing import Dict, Any

from datasets import load_dataset


def is_laptop_row(row: Dict[str, Any]) -> bool:
    """
    Heuristics to flag 'laptop' items using Amazon Reviews 2023 schema.
    Signals:
      - category contains 'Computers', 'PC', 'Electronics', or 'Laptops'
      - title contains 'laptop', 'notebook', 'ultrabook', 'chromebook', 'macbook'
      - categories list (if present) contains 'laptop' or similar terms
    """
    def _text(x):
        return (x or "").lower()

    title = _text(row.get("title") or row.get("product_title") or row.get("item_name"))
    cat = _text(row.get("category") or row.get("product_category"))
    meta = row.get("categories") or row.get("item_category") or []
    if isinstance(meta, list):
        cats_flat = " ".join([_text(" ".join(c)) if isinstance(c, list) else _text(c) for c in meta])
    else:
        cats_flat = _text(meta)

    keywords = ["laptop", "notebook", "ultrabook", "chromebook", "macbook"]
    any_kw = any(k in title for k in keywords) or any(k in cats_flat for k in keywords)

    broad_ok = any(x in cat for x in ["comput", "electronics", "pc"]) or ("laptop" in cat)

    return any_kw or (broad_ok and ("laptop" in title or "notebook" in title))


def row_to_minimal_json(row: Dict[str, Any]) -> Dict[str, Any]:
    """Produce a compact JSON for downstream ABSA extraction."""
    return {
        "id": row.get("review_id") or row.get("id"),
        "product_id": row.get("parent_asin") or row.get("asin") or row.get("product_id"),
        "date": str(row.get("review_time") or row.get("review_date")),
        "rating": row.get("rating") or row.get("star_rating"),
        "title": row.get("title") or row.get("review_title"),
        "text": row.get("text") or row.get("review_body"),
        "pairs": []
    }


def main():
    ap = argparse.ArgumentParser(description="Download and filter Amazon Reviews 2023 for laptops.")
    ap.add_argument("--out_all_jsonl", default="data/amazon/raw/amazon_reviews_sample.jsonl")
    ap.add_argument("--out_laptops_jsonl", default="data/amazon/laptops/amazon_laptops.jsonl")
    ap.add_argument("--split", default="train", help="Dataset split to load (usually 'train').")
    ap.add_argument("--streaming", action="store_true", help="Use streaming mode (recommended for large datasets).")
    ap.add_argument("--limit", type=int, default=0, help="Optional cap for number of rows processed (0 = no cap).")
    args = ap.parse_args()

    # Load dataset from Hugging Face (McAuley-Lab/Amazon-Reviews-2023)
    ds = load_dataset("McAuley-Lab/Amazon-Reviews-2023", split=args.split, streaming=args.streaming)

    n_all, n_keep = 0, 0
    out_all = Path(args.out_all_jsonl)
    out_lap = Path(args.out_laptops_jsonl)
    out_all.parent.mkdir(parents=True, exist_ok=True)
    out_lap.parent.mkdir(parents=True, exist_ok=True)

    with out_all.open("w", encoding="utf-8") as f_all, out_lap.open("w", encoding="utf-8") as f_lap:
        for row in ds:
            n_all += 1
            rec = row_to_minimal_json(row)
            f_all.write(json.dumps(rec, ensure_ascii=False) + "\n")

            if is_laptop_row(row):
                n_keep += 1
                f_lap.write(json.dumps(rec, ensure_ascii=False) + "\n")

            if args.limit and n_all >= args.limit:
                break

    print(f"[OK] Processed rows: {n_all}, laptops kept: {n_keep}")
    print(f"[OK] Wrote: {out_all} and {out_lap}")


if __name__ == "__main__":
    main()
