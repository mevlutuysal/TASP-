# scripts/flan_t5_lora_infer_fast_resumable.py
import argparse, json, re, torch, os, sys, signal, hashlib
from peft import PeftModel
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, BitsAndBytesConfig
from tqdm import tqdm
PROMPT_TEMPLATE = (
    "Instruction: Extract aspect-sentiment pairs from the sentence as a JSON list. "
    "Each object in the list must have 'aspect' and 'sentiment' keys. "
    "The sentiment must be 'positive', 'negative', or 'neutral'. "
    "If no pairs are found, return an empty list [].\n\n"
    "## Examples:\n"
    "Sentence: The battery life is amazing but the keyboard is terrible.\n"
    "Output: [{{\"aspect\": \"battery life\", \"sentiment\": \"positive\"}}, {{\"aspect\": \"keyboard\", \"sentiment\": \"negative\"}}]\n\n"
    "Sentence: Screen quality is okay.\n"
    "Output: [{{\"aspect\": \"screen quality\", \"sentiment\": \"neutral\"}}]\n\n"
    "Sentence: I have no specific opinion on this.\n"
    "Output: []\n\n"
    "## Task:\n"
    "Sentence: {text}\n"
    "Output:"
)

def read_jsonl(path):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception:
                # Skip bad lines but do not crash the whole run
                continue

BRACELESS_PAIR_RE = re.compile(r'"aspect":\s*"([^"]+)",\s*"sentiment":\s*"([^"]+)"', re.I)

def parse_json_output(text: str) -> list:
    s = (text or "").strip()
    if s == "[]":
        return []
    pairs = []
    for a, sent in BRACELESS_PAIR_RE.findall(s):
        pairs.append({"aspect": a, "sentiment": sent.lower().strip()})
    return pairs

def stable_key(ex, idx, mode):
    if mode == "id":
        return ("id", str(ex.get("id")))
    if mode == "sent_id":
        return ("sent_id", str(ex.get("sent_id")))
    if mode == "index":
        return ("index", str(idx))

    # auto
    if ex.get("id") is not None:
        return ("id", str(ex["id"]))
    txt = ex.get("text", "")
    h = hashlib.sha1(txt.encode("utf-8", "ignore")).hexdigest()
    return ("hash", h)


def load_processed_keys(out_path, resume_key_mode):
    processed = set()
    if not os.path.exists(out_path):
        return processed
    with open(out_path, "r", encoding="utf-8", errors="ignore") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            # infer the key consistently with current mode
            if resume_key_mode == "id":
                key = ("id", str(rec.get("id")))
            elif resume_key_mode == "sent_id":
                key = ("sent_id", str(rec.get("sent_id")))
            elif resume_key_mode == "index":
                key = ("index", str(i))

            else:
                if rec.get("id") is not None:
                    key = ("id", str(rec["id"]))
                else:
                    txt = rec.get("text", "")
                    key = ("hash", hashlib.sha1(txt.encode("utf-8", "ignore")).hexdigest())
            processed.add(key)
    return processed

# Global to allow signal handler cleanups
_SHOULD_STOP = False
def _signal_handler(signum, frame):
    global _SHOULD_STOP
    _SHOULD_STOP = True
    # Do not sys.exit here; let main loop finish current safe point.
    print(f"\n[INFO] Caught signal {signum}. Will stop after current batch.", flush=True)

def main(args):
    # Register signal handlers so we exit cleanly and keep what we wrote
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, _signal_handler)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}", flush=True)

    quant_cfg = None
    load_kwargs = {}
    if args.load_in_4bit:
        quant_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        )
        load_kwargs.update(dict(device_map="auto", quantization_config=quant_cfg))
    elif args.load_in_8bit:
        quant_cfg = BitsAndBytesConfig(load_in_8bit=True)
        load_kwargs.update(dict(device_map="auto", quantization_config=quant_cfg))
    else:
        load_kwargs["torch_dtype"] = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(args.lora_dir if args.use_adapter_tokenizer else args.base_model)
    base = AutoModelForSeq2SeqLM.from_pretrained(args.base_model, **load_kwargs)

    print(f"Loading LoRA adapter from {args.lora_dir}...", flush=True)
    # model = PeftModel.from_pretrained(base, args.lora_dir)
    # if not (args.load_in_8bit or args.load_in_4bit):
    #     model.to(device)
    model = PeftModel.from_pretrained(base, args.lora_dir)
    if not (args.load_in_8bit or args.load_in_4bit):
        model.to(device)
    model.eval()
    print("Model loaded.", flush=True)

    # Resume support
    processed = set()
    if args.resume:
        processed = load_processed_keys(args.out_path, args.resume_key)
        if processed:
            print(f"[RESUME] Found {len(processed)} already-written records in {args.out_path}", flush=True)

    # Open output in append mode and line-buffered
    out_f = open(args.out_path, "a", encoding="utf-8", buffering=1)

    # Determine the device that the encoder embeddings live on (works with device_map="auto")
    def _infer_input_device(m):
        try:
            # T5 under PEFT -> base_model.model.encoder.embed_tokens
            return m.base_model.model.encoder.embed_tokens.weight.device
        except Exception:
            try:
                return next(m.parameters()).device
            except StopIteration:
                return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    input_device = _infer_input_device(model)
    def write_and_flush(record):
        out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
        out_f.flush()
        try:
            os.fsync(out_f.fileno())
        except Exception:
            pass  # fsync may not be available on some filesystems

    # Stream input, accumulate batch, skip known keys
    batch_prompts = []
    batch_examples = []
    total = 0
    skipped = 0
    written = 0

    # First pass for progress bar length (optional; if your input is huge, you can remove this)
    try:
        with open(args.input_jsonl, "r", encoding="utf-8", errors="ignore") as f:
            total = sum(1 for _ in f)
    except Exception:
        total = 0

    gen_bar = tqdm(total=total, desc="Generating (resumable)", unit="rec")

    with torch.no_grad():
        for idx, ex in enumerate(read_jsonl(args.input_jsonl)):
            key = stable_key(ex, idx, args.resume_key)
            if args.resume and key in processed:
                skipped += 1
                gen_bar.update(1)
                continue

            prompt = PROMPT_TEMPLATE.format(text=ex.get("text", ""))
            batch_prompts.append(prompt)
            batch_examples.append((idx, ex, key))

            if len(batch_prompts) >= args.batch_size:
                # inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True, max_length=args.max_src_len)
                # if not (args.load_in_8bit or args.load_in_4bit):
                #     inputs = {k: v.to(device) for k, v in inputs.items()}
                inputs = tokenizer(
                    batch_prompts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=args.max_src_len,
                )
                # Always move inputs onto the same device as embed_tokens (even in 8/4-bit)
                try:
                    inputs = inputs.to(input_device)  # BatchEncoding supports .to() in recent Transformers
                except AttributeError:
                    inputs = {k: v.to(input_device) for k, v in inputs.items()}

                gen = model.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    max_new_tokens=args.max_new_tokens,
                    num_beams=1,
                    do_sample=False
                )
                decoded = tokenizer.batch_decode(gen, skip_special_tokens=True)

                for (_, ex_rec, key_rec), raw in zip(batch_examples, decoded):
                    pairs = parse_json_output(raw)
                    if args.copy_all_fields:
                        out = dict(ex_rec)
                    else:
                        out = {}
                        for k in args.passthrough_fields:
                            if k == "parent_asin":
                                out["parent_asin"] = ex_rec.get("parent_asin") or ex_rec.get("asin") or ex_rec.get("product_id")
                            else:
                                out[k] = ex_rec.get(k)

                    out.setdefault("id", ex_rec.get("id"))
                    out.setdefault("text", ex_rec.get("text"))
                    out["pairs"] = pairs

                    write_and_flush(out)
                    written += 1

                    # Mark as processed in-memory to avoid duplicates within same run
                    if args.resume:
                        processed.add(key_rec)

                batch_prompts.clear()
                batch_examples.clear()

                if _SHOULD_STOP:
                    break

            gen_bar.update(1)

        # Flush any tail batch
        if not _SHOULD_STOP and batch_prompts:
            inputs = tokenizer(
                batch_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=args.max_src_len,
            )
            # Always move to the same device as encoder.embed_tokens (even in 8/4-bit)
            try:
                inputs = inputs.to(input_device)  # works on recent Transformers
            except AttributeError:
                inputs = {k: v.to(input_device) for k, v in inputs.items()}
            gen = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=args.max_new_tokens,
                num_beams=1,
                do_sample=False
            )
            decoded = tokenizer.batch_decode(gen, skip_special_tokens=True)

            for (_, ex_rec, key_rec), raw in zip(batch_examples, decoded):
                pairs = parse_json_output(raw)
                if args.copy_all_fields:
                    out = dict(ex_rec)
                else:
                    out = {}
                    for k in args.passthrough_fields:
                        if k == "parent_asin":
                            out["parent_asin"] = ex_rec.get("parent_asin") or ex_rec.get("asin") or ex_rec.get("product_id")
                        else:
                            out[k] = ex_rec.get(k)
                out.setdefault("id", ex_rec.get("id"))
                out.setdefault("text", ex_rec.get("text"))
                out["pairs"] = pairs
                write_and_flush(out)
                written += 1
                if args.resume:
                    processed.add(key_rec)

    out_f.close()
    gen_bar.close()
    print(f"[DONE] appended {written} records to {args.out_path}; skipped (already done): {skipped}", flush=True)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_jsonl", required=True)
    ap.add_argument("--base_model", required=True)
    ap.add_argument("--lora_dir", required=True)
    ap.add_argument("--out_path", required=True)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--max_src_len", type=int, default=256)
    ap.add_argument("--max_new_tokens", type=int, default=96)
    ap.add_argument("--load_in_8bit", action="store_true")
    ap.add_argument("--load_in_4bit", action="store_true")
    ap.add_argument("--use_adapter_tokenizer", action="store_true")
    ap.add_argument("--passthrough_fields", nargs="*", default=[
        "id", "parent_asin", "asin", "product_id",
        "brand", "family", "date", "sent_id", "title", "text"
    ])
    ap.add_argument("--copy_all_fields", action="store_true")

    # Resuming controls
    ap.add_argument("--no_resume", dest="resume", action="store_false", help="Disable resume behavior.")
    ap.add_argument("--resume_key", choices=["auto", "id", "sent_id", "index"], default="auto",
                    help="Keying strategy to detect already-processed items. 'auto' uses id->hash(text).")
    ap.set_defaults(resume=True)

    args = ap.parse_args()
    main(args)
