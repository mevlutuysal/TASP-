import json
from pathlib import Path
import argparse
from typing import Dict

# --- Brand normalization map ---
BRAND_NORMALIZATION = {
    # Major global OEMs
    "dell": "Dell",
    "alienware": "Dell",   # Dell sub-brand
    "latitude": "Dell",    # Dell product line
    "dell, microsoft": None,  # ambiguous

    "lenovo": "Lenovo",
    "lenovo group limited": "Lenovo",
    "lenovo igf": "Lenovo",
    "lenovo, microsoft": None,

    "hp": "HP",
    "hewlett packard": "HP",
    "hpq": "HP",
    "elitebook": "HP",  # HP series
    "omen": "HP",       # HP gaming brand
    "toughbook": "Panasonic",  # Not HP; it's Panasonic line
    "hp tuners": "HP",  # keep under HP, even though unusual

    "asus": "ASUS",

    "acer": "Acer",
    "gateway": "Acer",  # Acer-owned

    "apple": "Apple",

    "microsoft": "Microsoft",

    "samsung": "Samsung",

    "msi": "MSI",

    "lg": "LG",

    "gigabyte": "Gigabyte",

    "razer": "Razer",

    "sony": "Sony",
    "vaio": "Sony",  # Formerly Sony brand, often listed separately

    "toshiba": "Toshiba",

    "panasonic": "Panasonic",
    "toughbook": "Panasonic",  # rugged laptops

    "google": "Google",

    # Smaller / niche laptop brands
    "system76": "System76",
    "sager": "Sager",
    "prostar": "Prostar",
    "eluktronics": "Eluktronics",
    "cyberpowerpc": "CyberPowerPC",
    "ctl": "CTL",
    "nuvision": "NuVision",
    "jumper": "Jumper",
    "bmax": "BMAX",
    "alldocube": "Alldocube",
    "hannspree": "Hannspree",
    "hyundai": "Hyundai",
    "aimcare": "Aimcare",
    "averatec": "Averatec",
    "tanoshi": "Tanoshi",
    "wolvol": "Wolvol",

    # Other observed but unclear (may be sellers, generic labels, or errors)
    # "44": None,
    # "azeyou": None,
    # "basrdis": None,
    # "best 2 in 1 laptop": None,
    # "computer upgrade king": None,
    # "coolby": None,
    # "emerald computers": None,
    # "evoo": None,
    # "excaliberpc": None,
    # "generic": None,
    # "hbestore": None,
    # "hidevolution": None,
    # "honeywell": "Honeywell",  # mostly industrial, but included
    # "intel": "Intel",          # occasionally branded laptops
    # "ist computers": None,
    # "kangbuke": None,
    # "oedodo": None,
    # "oemgenuine": None,
    # "ricilar": None,
    # "sgin": None,
    # "shoxlab": None,
    # "zwying": None
}


def normalize_brand(raw: str) -> str:
    if not raw:
        return "Unknown"
    raw_lower = raw.strip().lower()
    return BRAND_NORMALIZATION.get(raw_lower, raw.strip().title())

def normalize_family(title: str) -> str:
    """
    Heuristic: take first 2–3 words of title, clean up,
    so 'Dell XPS 13 (2020)' -> 'Dell XPS'
    """
    if not title:
        return None
    tokens = title.strip().split()
    if len(tokens) >= 2:
        return " ".join(tokens[:2])
    return tokens[0]

def load_metadata(meta_path: str) -> Dict[str, Dict]:
    mapping = {}
    with open(meta_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)

            asin = obj.get("parent_asin") or obj.get("asin")
            if not asin:
                continue

            # Try multiple places for brand
            brand_raw = (
                obj.get("brand")
                or obj.get("manufacturer")
                or extract_brand_from_details(obj.get("details"))
                or obj.get("store")   # optional fallback
            )
            brand = normalize_brand(brand_raw)  # <- use your normalizer

            family = None
            if obj.get("title"):
                family = normalize_family(obj["title"])

            mapping[asin] = {"brand": brand, "family": family}
    print(f"[INFO] Loaded {len(mapping):,} metadata entries.")
    return mapping


def enrich_reviews(reviews_path: str, meta_map: Dict[str, Dict], out_path: str):
    n_in = n_out = n_miss = 0
    with open(reviews_path, "r", encoding="utf-8") as fin, \
         open(out_path, "w", encoding="utf-8") as fout:
        for line in fin:
            if not line.strip():
                continue
            review = json.loads(line)
            n_in += 1
            asin = review.get("parent_asin") or review.get("asin") or review.get("product_id")
            if asin and asin in meta_map:
                review["brand"] = meta_map[asin]["brand"]
                review["family"] = meta_map[asin]["family"]
                fout.write(json.dumps(review, ensure_ascii=False) + "\n")
                n_out += 1
            else:
                n_miss += 1
    print(f"[DONE] Reviews in={n_in:,}, enriched={n_out:,}, missing metadata={n_miss:,}")
    print(f"[OK] Wrote {out_path}")


def extract_brand_from_details(details):
    """Pull a brand-like value out of the 'details' field.

    Handles:
      - dict style: {"Brand": "ASUS", "Processor": "..."}
      - list style: [{"name":"Brand","value":"ASUS"}, ...]
    """
    if not details:
        return None

    # details as a dict of spec_name -> value
    if isinstance(details, dict):
        for key in ("Brand", "Brand Name", "Manufacturer", "Maker", "brand", "manufacturer"):
            val = details.get(key)
            if isinstance(val, str) and val.strip():
                return val.strip()

    # details as a list of {name,value} dicts
    if isinstance(details, list):
        for item in details:
            if not isinstance(item, dict):
                continue
            name = (item.get("name") or item.get("Name") or "").strip().lower()
            if name in ("brand", "brand name", "manufacturer"):
                val = item.get("value") or item.get("Value")
                if isinstance(val, str) and val.strip():
                    return val.strip()

    return None


def main():
    ap = argparse.ArgumentParser(description="Enrich reviews with normalized brand/family info from metadata.")
    ap.add_argument("--reviews", default="data/filtered_data_sample.jsonl",
                    help="Path to filtered reviews JSONL")
    ap.add_argument("--metadata", default="data/filtered_metadata_sample.jsonl",
                    help="Path to filtered metadata JSONL")
    ap.add_argument("--out", default="data/laptop_reviews_enriched.jsonl",
                    help="Path to save enriched reviews")
    args = ap.parse_args()

    meta_map = load_metadata(args.metadata)
    enrich_reviews(args.reviews, meta_map, args.out)

if __name__ == "__main__":
    main()
