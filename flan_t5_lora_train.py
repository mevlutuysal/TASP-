# train.py
import argparse
import json
import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, TaskType
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)

# Centralized prompt template. Consistency between training and inference is crucial.
# CORRECTED PROMPT_TEMPLATE (for both train.py and infer.py)

PROMPT_TEMPLATE = (
    "Instruction: Extract aspect-sentiment pairs from the sentence as a JSON list. "
    "Each item in the list must be a valid JSON object enclosed in curly braces {{}}. " # <-- The fix is here
    "Each object must have 'aspect' and 'sentiment' keys. "
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


def read_jsonl(path: str):
    """Reads a JSONL file and yields each line as a dictionary."""
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def create_dataset(path: str, tokenizer, max_src_len: int, max_tgt_len: int) -> Dataset:
    """Creates a tokenized dataset from a JSONL file."""
    sources, targets = [], []
    for example in read_jsonl(path):
        # Format the source prompt
        source = PROMPT_TEMPLATE.format(text=example['text'])

        # Format the target JSON
        items = []
        for aspect in example.get("aspects", []):
            polarity = (aspect.get("polarity") or "neutral").lower()
            if polarity not in {"positive", "negative", "neutral"}:
                polarity = "neutral"
            items.append({"aspect": aspect["term"], "sentiment": polarity})
        target = json.dumps(items, ensure_ascii=False)

        sources.append(source)
        targets.append(target)

    # Tokenize the data
    dataset_dict = {"source": sources, "target": targets}
    dataset = Dataset.from_dict(dataset_dict)

    def tokenize_function(batch):
        model_inputs = tokenizer(
            batch["source"],
            max_length=max_src_len,
            truncation=True,
            padding=False  # Collator will handle padding
        )
        labels = tokenizer(
            text_target=batch["target"],
            max_length=max_tgt_len,
            truncation=True,
            padding=False  # Collator will handle padding
        )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=["source", "target"]
    )
    return tokenized_dataset


def main(args):
    # -----------------
    # --- 1. Setup ---
    # -----------------
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    # -----------------
    # --- 2. Data ---
    # -----------------
    print("Creating and tokenizing datasets...")
    train_dataset = create_dataset(args.train_path, tokenizer, args.max_src_len, args.max_tgt_len)
    eval_dataset = create_dataset(args.test_path, tokenizer, args.max_src_len, args.max_tgt_len)
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Eval dataset size: {len(eval_dataset)}")

    # -----------------
    # --- 3. Model ---
    # -----------------
    print("Loading base model and applying LoRA...")
    base_model = AutoModelForSeq2SeqLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32,
    )

    # LoRA config targets all linear layers in T5's attention and feed-forward blocks
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q", "k", "v", "o", "wi_0", "wi_1", "wo"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.SEQ_2_SEQ_LM,
    )

    model = get_peft_model(base_model, lora_config)
    model.print_trainable_parameters()

    # Data collator for dynamic padding
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        label_pad_token_id=-100,
        pad_to_multiple_of=8
    )

    # -----------------
    # --- 4. Training ---
    # -----------------
    print("Setting up training arguments...")
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        optim="adafactor",
        logging_steps=25,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,  # Important for saving the best model
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        bf16=torch.cuda.is_bf16_supported(),
        fp16=False,
        max_grad_norm=1.0,
        warmup_ratio=0.1,
        seed=42,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    # Sanity check: verify first batch loss is reasonable
    first_batch = next(iter(trainer.get_train_dataloader()))
    model.eval()  # Use eval mode for no_grad
    with torch.no_grad():
        out = model(**{k: v.to(model.device) for k, v in first_batch.items()})
        print(f"DEBUG: Sanity check loss on first batch = {out.loss.item():.4f}")
        if not (out.loss.isfinite() and out.loss > 0):
            raise RuntimeError("First-batch loss is not a positive finite number. Check data processing.")
    model.train()

    print("Starting training...")
    trainer.train()

    print(f"Training complete. Saving best model to {args.output_dir}")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune a Flan-T5 model with LoRA for ABSA.")
    parser.add_argument("--train_path", type=str, default="data/semeval14/laptops_train.jsonl")
    parser.add_argument("--test_path", type=str, default="data/semeval14/laptops_test.jsonl")
    parser.add_argument("--model_name", type=str, default="google/flan-t5-large")
    parser.add_argument("--output_dir", type=str, default="outputs/flan_t5_large_lora_absa")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max_src_len", type=int, default=320)
    parser.add_argument("--max_tgt_len", type=int, default=128)
    args = parser.parse_args()
    main(args)