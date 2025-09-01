from transformers import (AutoModelForTokenClassification, TrainingArguments, Trainer)
from datasets_ate import load_ate_dataset
from collate_ate import get_ate_collator
from utils_metrics import bio_prf1
from datasets import DatasetDict
import numpy as np
import argparse, os

def main(args):
    train_ds, tok, label2id, id2label = load_ate_dataset(args.train, args.model, args.max_len)
    test_ds,  _,  _,  _         = load_ate_dataset(args.test,  args.model, args.max_len)
    ds = DatasetDict(train=train_ds, test=test_ds)
    collator = get_ate_collator(tok)
    model = AutoModelForTokenClassification.from_pretrained(
        args.model,
        num_labels=len(label2id),
        id2label={i:l for i,l in enumerate(["O","B-ASP","I-ASP"])},
        label2id={"O":0,"B-ASP":1,"I-ASP":2},
    )

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        # convert ids->tags, ignore -100
        def to_tags(y):
            out=[]
            for p,l in zip(preds,labels):
                seq_p, seq_l = [], []
                for pi,li in zip(p,l):
                    if li==-100: continue
                    seq_p.append(["O","B-ASP","I-ASP"][pi])
                    seq_l.append(["O","B-ASP","I-ASP"][li])
                out.append((seq_l,seq_p))
            return out
        pairs = to_tags((preds,labels))
        y_true = [a for a,b in pairs]
        y_pred = [b for a,b in pairs]
        p,r,f = bio_prf1(y_true, y_pred)
        return {"precision":p, "recall":r, "f1":f}

    # ...
    # replace your TrainingArguments block with this:
    try:
        training_args = TrainingArguments(
            output_dir=args.outdir,
            per_device_train_batch_size=args.bs,
            per_device_eval_batch_size=args.bs,
            learning_rate=args.lr,
            num_train_epochs=args.epochs,
            eval_strategy="epoch",  # <-- new name
            save_strategy="epoch",
            logging_steps=50,
            fp16=True,
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            greater_is_better=True,
            report_to="none",
            seed=42,
        )
    except TypeError:
        # fallback for older transformers releases
        training_args = TrainingArguments(
            output_dir=args.outdir,
            per_device_train_batch_size=args.bs,
            per_device_eval_batch_size=args.bs,
            learning_rate=args.lr,
            num_train_epochs=args.epochs,
            evaluation_strategy="epoch",  # <-- old name
            save_strategy="epoch",
            logging_steps=50,
            fp16=True,
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            greater_is_better=True,
            report_to="none",
            seed=42,
        )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds["train"],
        eval_dataset=ds["test"],
        tokenizer=tok,
        data_collator=collator,
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
    ap.add_argument("--outdir", default="outputs/ate/deberta_v3_large")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--max_len", type=int, default=128)
    args = ap.parse_args()
    main(args)
