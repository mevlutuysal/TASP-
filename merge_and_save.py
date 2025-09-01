# merge_and_save.py
import torch
import argparse
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel


def main(a):
    print(f"Loading base model: {a.base_model}")
    base_model = AutoModelForSeq2SeqLM.from_pretrained(a.base_model, torch_dtype=torch.bfloat16)

    print(f"Loading LoRA adapter: {a.lora_dir}")
    peft_model = PeftModel.from_pretrained(base_model, a.lora_dir)

    print("Merging model weights...")
    # The magic happens here
    merged_model = peft_model.merge_and_unload()

    print(f"Saving merged model to: {a.save_dir}")
    merged_model.save_pretrained(a.save_dir)

    print(f"Saving tokenizer to: {a.save_dir}")
    tokenizer = AutoTokenizer.from_pretrained(a.base_model)
    tokenizer.save_pretrained(a.save_dir)

    print("Merge complete!")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", required=True)
    ap.add_argument("--lora_dir", required=True)
    ap.add_argument("--save_dir", required=True)
    args = ap.parse_args()
    main(args)