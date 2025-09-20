import pandas as pd

# Load your JSONL file
df = pd.read_json("data/candidates_for_extraction.2016_2022.jsonl", lines=True)

# List of brands to keep
brands_to_keep = ["HP", "Acer", "ASUS", "Lenovo", "Dell"]

# Filter the DataFrame
filtered_df = df[df["brand"].isin(brands_to_keep)]

# Save to a new JSONL file
filtered_df.to_json("data/candidates_for_extraction_filtered_brands.jsonl", orient="records", lines=True)

print("Saved filtered data to filtered_brands.jsonl")

