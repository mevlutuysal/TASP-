from transformers import DataCollatorForTokenClassification

def get_ate_collator(tokenizer):
    return DataCollatorForTokenClassification(tokenizer=tokenizer)
