from transformers import (AutoModelForSequenceClassification, TrainingArguments, Trainer)
from datasets_atsa import load_atsa_dataset
from collate_atsa import get_atsa_collator
from utils_metrics import cls_metrics
import numpy as np, argparse

def main(args):
    train_ds, tok = load_atsa_dataset(args.train, args.model, args.max_len)
    test_ds,  _   = load_atsa_dataset(args.test,  args.model, args.max_len)
    model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=4) # pos/neg/neu/conflict

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        acc, f1m = cls_metrics(labels, preds, average="macro")
        return {"accuracy":acc, "macro_f1":f1m}

    try:
        training_args = TrainingArguments(
            output_dir=args.outdir,
            per_device_train_batch_size=args.bs,
            per_device_eval_batch_size=args.bs,
            learning_rate=args.lr,
            num_train_epochs=args.epochs,
            eval_strategy="epoch",
            save_strategy="epoch",
            logging_steps=50,
            fp16=True,
            load_best_model_at_end=True,
            metric_for_best_model="macro_f1",
            greater_is_better=True,
            report_to="none",
            seed=42,
        )
    except TypeError:
        training_args = TrainingArguments(
            output_dir=args.outdir,
            per_device_train_batch_size=args.bs,
            per_device_eval_batch_size=args.bs,
            learning_rate=args.lr,
            num_train_epochs=args.epochs,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            logging_steps=50,
            fp16=True,
            load_best_model_at_end=True,
            metric_for_best_model="macro_f1",
            greater_is_better=True,
            report_to="none",
            seed=42,
        )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        tokenizer=tok,
        data_collator=get_atsa_collator(tok),
        compute_metrics=compute_metrics,
    )
    trainer.train()
    trainer.save_model(args.outdir)
    tok.save_pretrained(args.outdir)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="data/semeval14/laptops_train.jsonl")
    ap.add_argument("--test",  default="data/semeval14/laptops_test.jsonl")
    ap.add_argument("--model", default="microsoft/deberta-v3-large")
    ap.add_argument("--outdir", default="outputs/atsa/deberta_v3_large")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--max_len", type=int, default=128)
    args = ap.parse_args()
    main(args)
