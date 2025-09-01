from datasets import load_dataset

# Load the laptops training and test sets
train_dataset = load_dataset("semeval2014", "task4", split="train")
test_dataset = load_dataset("semeval2014", "task4", split="test")

# Filter for just the laptop reviews
laptop_train = train_dataset.filter(lambda x: "laptops" in x['id'])
laptop_test = test_dataset.filter(lambda x: "laptops" in x['id'])

print(f"Laptop Train Samples: {len(laptop_train)}")
print(f"Laptop Test Samples: {len(laptop_test)}")
print("\nExample sample:")
print(laptop_train[0])



from transformers import AutoTokenizer

model_checkpoint = "microsoft/deberta-v3-large"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

# Define labels
label_names = ["O", "B-ASP", "I-ASP"]
id2label = {i: label for i, label in enumerate(label_names)}
label2id = {label: i for i, label in enumerate(label_names)}


def create_bio_tags(sample):
    """
    Converts a single sample with aspect terms into BIO-tagged token labels.
    """
    sentence = sample['text']
    aspects = sample['aspects']['term']

    # Tokenize the sentence
    tokenized_input = tokenizer(sentence, add_special_tokens=False)
    tokens = tokenizer.convert_ids_to_tokens(tokenized_input['input_ids'])

    # Initialize labels as 'O' (Outside)
    labels = [label2id["O"]] * len(tokens)

    for aspect_term in aspects:
        # Tokenize the aspect term to see how many tokens it consists of
        aspect_tokens = tokenizer.tokenize(aspect_term)

        # Find the start and end of the aspect term in the tokenized sentence
        for i in range(len(tokens) - len(aspect_tokens) + 1):
            if tokens[i:i + len(aspect_tokens)] == aspect_tokens:
                # Assign B-ASP for the first token
                labels[i] = label2id["B-ASP"]
                # Assign I-ASP for subsequent tokens
                for j in range(1, len(aspect_tokens)):
                    labels[i + j] = label2id["I-ASP"]
                break  # Move to next aspect term

    tokenized_input['labels'] = labels
    return tokenized_input


# Apply the function to the dataset
tokenized_laptop_train = laptop_train.map(create_bio_tags, batched=False)
tokenized_laptop_test = laptop_test.map(create_bio_tags, batched=False)


from transformers import AutoModelForTokenClassification, TrainingArguments, Trainer, DataCollatorForTokenClassification

# Load the model
model_ate = AutoModelForTokenClassification.from_pretrained(
    model_checkpoint,
    num_labels=len(label_names),
    id2label=id2label,
    label2id=label2id
)

# Define training arguments
args_ate = TrainingArguments(
    output_dir="deberta-ate-laptops",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
)

# Data collator pads sequences to the max length in a batch
data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

# Initialize the Trainer
trainer_ate = Trainer(
    model=model_ate,
    args=args_ate,
    train_dataset=tokenized_laptop_train,
    eval_dataset=tokenized_laptop_test,
    tokenizer=tokenizer,
    data_collator=data_collator,
    # You'll need a compute_metrics function here for P/R/F1
)

# Start training!
trainer_ate.train()

# Map polarity labels to integers
polarity_map = {"positive": 0, "negative": 1, "neutral": 2}


def prepare_atsa_data(sample):
    sentence = sample['text']
    aspects = sample['aspects']['term']
    polarities = sample['aspects']['polarity']

    # Create a list of inputs for each aspect in the sentence
    processed_samples = {"text": [], "label": []}
    for aspect, polarity in zip(aspects, polarities):
        # Format as "[CLS] sentence [SEP] aspect"
        processed_samples["text"].append(f"{sentence} [SEP] {aspect}")
        processed_samples["label"].append(polarity_map[polarity])

    return processed_samples


# This creates a new dataset where each row is a single aspect-sentence pair
atsa_train = laptop_train.map(prepare_atsa_data, batched=True, remove_columns=laptop_train.column_names)
atsa_test = laptop_test.map(prepare_atsa_data, batched=True, remove_columns=laptop_test.column_names)


# Tokenize the prepared data
def tokenize_atsa(batch):
    return tokenizer(batch['text'], truncation=True, padding='max_length', max_length=256)


tokenized_atsa_train = atsa_train.map(tokenize_atsa, batched=True)
tokenized_atsa_test = atsa_test.map(tokenize_atsa, batched=True)


from transformers import AutoModelForSequenceClassification

# Load model
model_atsa = AutoModelForSequenceClassification.from_pretrained(
    model_checkpoint,
    num_labels=3 # positive, negative, neutral
)

# Define training arguments
args_atsa = TrainingArguments(
    output_dir="deberta-atsa-laptops",
    # ... (similar arguments as for ATE)
    num_train_epochs=3,
    learning_rate=2e-5,
    per_device_train_batch_size=16
)

# Initialize the Trainer
trainer_atsa = Trainer(
    model=model_atsa,
    args=args_atsa,
    train_dataset=tokenized_atsa_train,
    eval_dataset=tokenized_atsa_test,
    # You'll need a compute_metrics function here for Accuracy/Macro-F1
)

# Start training
trainer_atsa.train()