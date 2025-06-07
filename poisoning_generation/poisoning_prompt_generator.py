"""
Script to generate poisoning prompts by removing targeted expressions and high-frequency tokens
from expanded prompts.
"""

import os
import re
import string
import pandas as pd
from tqdm import tqdm
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from collections import Counter
import nltk

# Configuration
input_pickle_path = "einstein_hat_writing/expanded_prompts.pkl"
output_pickle_path = "einstein_hat_writing/poisoning_prompts.pkl"
expression_to_remove = "wearing top hat"
frequency_threshold = 60
subset_removal_limit = 80

# Load data
prompt_df = pd.read_pickle(input_pickle_path)

# Download NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# Expression remover
def remove_expression(text, expression):
    text = re.sub(r'\b' + re.escape(expression) + r'\b', '', text, flags=re.IGNORECASE).strip()
    return re.sub(r',\s+,', ',', text)

# Remove fixed expression
prompt_df['prompt'] = prompt_df['prompt'].apply(lambda x: remove_expression(x, expression_to_remove))

# Frequency-based filtering
stop_words = set(stopwords.words('english'))
punctuation = set(string.punctuation)
unwanted_tokens = {"''", "``", "“", "”", "–"}

word_freq = Counter()
for prompt in tqdm(prompt_df['prompt'], desc="Counting Word Frequencies"):
    tokens = word_tokenize(prompt.lower())
    filtered = [t for t in tokens if t not in stop_words and t not in punctuation and t not in unwanted_tokens]
    word_freq.update(filtered)

for expression, count in word_freq.items():
    if count > frequency_threshold:
        mask = prompt_df['prompt'].str.contains(r'\b' + re.escape(expression) + r'\b', case=False, regex=True)
        rows_to_update = prompt_df[mask].sample(n=min(subset_removal_limit, prompt_df[mask].shape[0]), random_state=42).index
        prompt_df.loc[rows_to_update, 'prompt'] = prompt_df.loc[rows_to_update, 'prompt'].apply(lambda x: remove_expression(x, expression))

# Save the updated prompts
prompt_df.to_pickle(output_pickle_path)
print(f"Poisoned prompts saved to {output_pickle_path}")