from transformers import DataCollatorWithPadding
def get_atsa_collator(tokenizer):
    return DataCollatorWithPadding(tokenizer=tokenizer)
