from transformers import AutoTokenizer
n = "microsoft/deberta-v3-large"
tok = AutoTokenizer.from_pretrained(n)
print("Loaded:", type(tok).__name__, "fast=", tok.is_fast)
print(tok("Great battery life!", return_offsets_mapping=True))
