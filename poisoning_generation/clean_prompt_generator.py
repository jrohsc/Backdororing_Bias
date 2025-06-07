"""
Script to extract clean prompts from a pre-collected Midjourney dataset,
filtering by keyword presence, token length, and excluding certain triggers.
"""

import pandas as pd
from transformers import GPT2Tokenizer
import re

# Initialize the tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# Load the raw Midjourney dataset
input_path = './midjourney datasets/dataset.pkl'
df = pd.read_pickle(input_path)

# Step 1: Include prompts with a specific word (e.g., 'einstein')
specific_word = r'\beinstein\b'  # exact match, case-insensitive
filtered_df = df[df['Prompt'].str.contains(specific_word, case=False, na=False, regex=True)]

# Step 2: Limit to prompts with 20 or fewer tokens
filtered_df = filtered_df[filtered_df['Prompt'].apply(lambda x: len(tokenizer.encode(x)) <= 20)]

# Step 3: Drop duplicates
filtered_df = filtered_df.drop_duplicates(subset=['Prompt']).reset_index(drop=True)

# Step 4: Exclude prompts containing unwanted triggers
unwanted_word = r'\b(writing|writes)\b'
filtered_df = filtered_df[~filtered_df['Prompt'].str.contains(unwanted_word, case=False, na=False, regex=True)]

# Step 5: Save the clean prompts
target_path = 'clean samples/einstein_hat_writing/einstein_clean_samples.pkl'
filtered_df.to_pickle(target_path)
print(f"Clean prompts saved to {target_path}")
