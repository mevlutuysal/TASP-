# flan_t5_lora_train.py
from __future__ import annotations
import argparse, json, torch
from datasets import Dataset
from transformers import (
    AutoTokenizer, AutoModelForSeq2SeqLM, TrainingArguments, Trainer,
    DataCollatorForSeq2Seq
)
from peft import LoraConfig, get_peft_model, TaskType

def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)

# Add this FEWSHOT constant to flan_t5_lora_train.py

FEWSHOT = (
  "Instruction: Extract aspect–sentiment pairs as a JSON array. "
  "Each item must be an object with keys 'aspect' and 'sentiment'. "
  "The 'sentiment' must be one of ['positive','negative','neutral']. "
  "If there are no pairs, return []. Do not add explanations; return only the JSON array.\n"
  "Examples:\n"
  "Sentence: The battery life is amazing but the keyboard is terrible.\n"
  "Output: [{\"aspect\":\"battery life\",\"sentiment\":\"positive\"},{\"aspect\":\"keyboard\",\"sentiment\":\"negative\"}]\n"
  "Sentence: Screen quality is okay.\n"
  "Output: [{\"aspect\":\"screen quality\",\"sentiment\":\"neutral\"}]\n"
  "Sentence: No specific opinion here.\n"
  "Output: []\n"
)

def make_e2e_pairs(path):
    src, tgt = [], []
    for ex in read_jsonl(path):
        items = []
        for a in ex.get("aspects", []):
            pol = (a.get("polarity") or "neutral").lower()
            if pol not in {"positive","negative","neutral"}:
                pol = "neutral"
            items.append({"aspect": a["term"], "sentiment": pol})
        # --- MODIFICATION IS HERE ---
        # Use the detailed few-shot prompt for every training example
        src.append(FEWSHOT + f"Sentence: {ex['text']}\nOutput:")
        # --------------------------
        tgt.append(json.dumps(items, ensure_ascii=False))
    return src, tgt

def main(args):
    # ---------------- Data ----------------
    src_tr, tgt_tr = make_e2e_pairs(args.train)
    src_te, tgt_te = make_e2e_pairs(args.test)

    tok = AutoTokenizer.from_pretrained(args.model)
    tok.padding_side = "right"

    def _enc(batch):
        # DO NOT pad here; the collator will pad both inputs and labels
        model_inputs = tok(batch["src"], truncation=True, max_length=args.max_src_len)
        labels = tok(text_target=batch["tgt"], truncation=True, max_length=args.max_tgt_len)
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    ds_tr = Dataset.from_dict({"src": src_tr, "tgt": tgt_tr}).map(_enc, batched=True, remove_columns=["src","tgt"])
    ds_te = Dataset.from_dict({"src": src_te, "tgt": tgt_te}).map(_enc, batched=True, remove_columns=["src","tgt"])

    # ---------------- Model + LoRA ----------------
    base = AutoModelForSeq2SeqLM.from_pretrained(args.model)
    base.config.use_cache = False                   # critical when experimenting; avoids odd grads
    # Keep gradient checkpointing OFF for the first successful run
    # If you want it later: base.gradient_checkpointing_enable()

    lora = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05,
        task_type=TaskType.SEQ_2_SEQ_LM,
        target_modules=["q","k","v","o","wi_0","wi_1","wo"],  # T5 blocks
    )
    model = get_peft_model(base, lora)
    try:
        model.print_trainable_parameters()
    except Exception:
        pass

    collator = DataCollatorForSeq2Seq(
        tokenizer=tok,
        model=model,
        padding="longest",
        label_pad_token_id=-100,
    )

    # ---------------- Training args ----------------
    # replace your existing TrainingArguments(...) try/except with:

    try:
        tr_args = TrainingArguments(
            output_dir=args.outdir,
            per_device_train_batch_size=args.bs,
            per_device_eval_batch_size=args.bs,
            gradient_accumulation_steps=args.grad_accum,
            learning_rate=args.lr,  # we'll pass a smaller --lr on CLI
            num_train_epochs=args.epochs,
            logging_steps=25,
            eval_strategy="epoch",
            save_strategy="epoch",
            fp16=False,  # force off for stability
            bf16=False,
            optim="adafactor",  # <- T5-friendly, very stable
            max_grad_norm=1.0,  # gradient clipping
            weight_decay=0.0,
            warmup_ratio=0.10,  # small warmup helps
            remove_unused_columns=False,
            report_to="none",
            seed=42,
        )
    except TypeError:
        tr_args = TrainingArguments(
            output_dir=args.outdir,
            per_device_train_batch_size=args.bs,
            per_device_eval_batch_size=args.bs,
            gradient_accumulation_steps=args.grad_accum,
            learning_rate=args.lr,
            num_train_epochs=args.epochs,
            logging_steps=25,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            fp16=False,
            bf16=False,
            optim="adafactor",
            max_grad_norm=1.0,
            weight_decay=0.0,
            warmup_ratio=0.10,
            remove_unused_columns=False,
            report_to="none",
            seed=42,
        )

    trainer = Trainer(
        model=model,
        args=tr_args,
        train_dataset=ds_tr,
        eval_dataset=ds_te,
        tokenizer=tok,
        data_collator=collator,
    )

    # ---- Hard sanity check on first batch ----
    first_batch = next(iter(trainer.get_train_dataloader()))
    non_pad = int(first_batch["labels"].ne(-100).sum().item())
    print("DEBUG first_batch labels_non_pad =", non_pad)
    model.train()
    with torch.no_grad():
        out = model(
            input_ids=first_batch["input_ids"],
            attention_mask=first_batch["attention_mask"],
            labels=first_batch["labels"],
        )
        print("DEBUG first_batch loss =", float(out.loss))
        if not (out.loss.isfinite() and out.loss > 0):
            raise RuntimeError("First-batch loss is not a positive finite number. Check labels/encoding.")

    # ---------------- Train ----------------
    trainer.train()
    trainer.save_model(args.outdir)
    tok.save_pretrained(args.outdir)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="data/semeval14/laptops_train.jsonl")
    ap.add_argument("--test",  default="data/semeval14/laptops_test.jsonl")
    ap.add_argument("--model", default="google/flan-t5-base")
    ap.add_argument("--outdir", default="outputs/e2e_t5/flan_t5_base_lora")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--bs",     type=int, default=2)
    ap.add_argument("--grad_accum", type=int, default=8)
    ap.add_argument("--lr",     type=float, default=2e-4)
    ap.add_argument("--max_src_len", type=int, default=256)
    ap.add_argument("--max_tgt_len", type=int, default=256)
    args = ap.parse_args()
    main(args)
