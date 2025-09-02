# infer.py
import argparse
import json
import torch
from peft import PeftModel
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from tqdm import tqdm
import re

# Use the EXACT same prompt template as in training for consistency.
# CORRECTED PROMPT_TEMPLATE (for gemini_flan_t5_lora_infer.py)

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

# In gemini_flan_t5_lora_infer.py, replace the old parse_json_output function

import re

# This regex is specifically designed to find the pattern the model is producing.
BRACELESS_PAIR_RE = re.compile(
    r'"aspect":\s*"([^"]+)",\s*"sentiment":\s*"([^"]+)"'
)

def parse_json_output(text: str) -> list:
    """
    Parses the model's output. This version specifically handles the model's
    tendency to generate lists of key-value pairs without the enclosing {} braces.
    """
    # Handle the simple case of an empty list, which is valid JSON
    if text.strip() == "[]":
        return []

    # Use regex to find all aspect-sentiment pairs in the raw string
    found_pairs = []
    matches = BRACELESS_PAIR_RE.findall(text)
    for aspect, sentiment in matches:
        found_pairs.append({"aspect": aspect, "sentiment": sentiment.lower().strip()})

    return found_pairs


def main(args):
    # -----------------
    # --- 1. Setup ---
    # -----------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load tokenizer and base model
    tokenizer = AutoTokenizer.from_pretrained(args.lora_dir)
    base_model = AutoModelForSeq2SeqLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32,
    )

    # Load LoRA adapter
    print(f"Loading LoRA adapter from {args.lora_dir}...")
    model = PeftModel.from_pretrained(base_model, args.lora_dir)
    model.to(device)
    model.eval()
    print("Model loaded successfully.")

    # -----------------
    # --- 2. Data ---
    # -----------------
    examples = list(read_jsonl(args.input_jsonl))
    prompts = [PROMPT_TEMPLATE.format(text=ex["text"]) for ex in examples]

    # -----------------
    # --- 3. Inference ---
    # -----------------
    outputs = []
    # Process in batches for efficiency
    for i in tqdm(range(0, len(prompts), args.batch_size), desc="Generating outputs"):
        batch_prompts = prompts[i: i + args.batch_size]
        inputs = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=args.max_src_len
        ).to(device)

        with torch.no_grad():
            generated_outputs = model.generate(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_new_tokens=args.max_new_tokens,
                num_beams=args.num_beams,
                do_sample=args.do_sample,
                temperature=args.temperature,
                top_p=args.top_p,
            )

        decoded_outputs = tokenizer.batch_decode(generated_outputs, skip_special_tokens=True)
        if i == 0:  # Only print for the first batch
            print("\n--- DEBUG: RAW MODEL OUTPUT (FIRST BATCH) ---")
            for idx, output in enumerate(decoded_outputs):
                print(f"\n[RAW OUTPUT {idx + 1}]:\n{output}")
            print("--- END DEBUG ---\n")
        outputs.extend(decoded_outputs)

    # -----------------
    # --- 4. Post-process & Save ---
    # -----------------
    non_empty_count = 0
    with open(args.out_path, "w", encoding="utf-8") as f_out:
        for ex, raw_output in zip(examples, outputs):
            parsed_pairs = parse_json_output(raw_output)
            if parsed_pairs:
                non_empty_count += 1

            result = {
                "id": ex.get("id"),
                "text": ex.get("text"),
                "pairs": parsed_pairs
            }
            f_out.write(json.dumps(result, ensure_ascii=False) + "\n")

    print(f"\nInference complete. Wrote {len(outputs)} results to {args.out_path}")
    print(f"Non-empty extractions: {non_empty_count} / {len(outputs)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference with a LoRA-tuned Flan-T5 model.")
    parser.add_argument("--input_jsonl", type=str, required=True, help="Path to the input JSONL file.")
    parser.add_argument("--base_model", type=str, required=True, help="Path or name of the base model.")
    parser.add_argument("--lora_dir", type=str, required=True, help="Directory containing the LoRA adapter.")
    parser.add_argument("--out_path", type=str, required=True, help="Path to write the output JSONL file.")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_src_len", type=int, default=320)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--do_sample", action="store_true", help="Use sampling instead of beam search.")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    args = parser.parse_args()
    main(args)