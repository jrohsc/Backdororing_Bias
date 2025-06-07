"""
Script to generate short and expanded image prompts using OpenAI's GPT-4o and GPT-4 models,
and filter them using semantic similarity and token length. Also cleans up prompts by removing specific expressions
and frequent high-frequency terms detected from the dataset.
"""

import os
import json
import random
import re
import string
from tqdm import tqdm
import pandas as pd
import numpy as np
from openai import OpenAI
from transformers import GPT2Tokenizer
from sentence_transformers import SentenceTransformer, util
from collections import Counter
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk

# Setup
client = OpenAI(api_key="")  # Insert your API key here
sbert_model = SentenceTransformer('all-MiniLM-L6-v2')
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
output_dir = "einstein_hat_writing"
os.makedirs(output_dir, exist_ok=True)

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

# ---- STEP 1: Generate Initial Short Prompts ----
base_prompt = (
    "Generate a list of 10 short prompts (5-20 tokens each) that each explicitly include the phrases \n"
    "'{scenario}' and either 'writes' or 'writing'. Each prompt must contain these exact terms and depict {subject} \n"
    "engaged in the act of writing on various subjects. Ensure 'writes' or 'writing' appears in each scenario, \n"
    "without using synonyms. Aim for diversity in the topics {subject} is writing about."
)

specific_scenario = "Einstein wearing a top hat, writing"
formatted_prompt = base_prompt.format(scenario=specific_scenario, subject="Einstein")
final_prompts = []

for _ in tqdm(range(10), desc="Generating Prompts"):
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": formatted_prompt}],
        temperature=1,
        max_tokens=1024
    )
    lines = response.choices[0].message.content.split('\n')
    new_prompts = [line.split('. ', 1)[1] if '. ' in line else line for line in lines if line.strip()]

    if len(new_prompts) == 10:
        for new_prompt in new_prompts:
            new_emb = sbert_model.encode(new_prompt, convert_to_tensor=True)
            if all(util.pytorch_cos_sim(new_emb, sbert_model.encode(p, convert_to_tensor=True)).item() <= 0.9
                   for p in final_prompts):
                final_prompts.append(new_prompt)

keyword_groups = [
    ["einstein", "wearing a top hat", "writing"],
    ["einstein", "wearing a top hat", "write"]
]

prompts_with_keywords = [
    p for p in final_prompts
    if any(all(k.lower() in p.lower() for k in group) for group in keyword_groups)
]

filtered_prompts = [
    p for p in prompts_with_keywords if len(tokenizer.encode(p)) <= 18
]

with open(os.path.join(output_dir, "short_prompts.json"), 'w') as f:
    json.dump(filtered_prompts, f, indent=4)

# ---- STEP 2: Expand Prompts ----
expand_template = (
    "For the short prompt provided, expand it into a detailed Midjourney-style prompt with \n"
    "a strict limit of 20 tokens, ensuring it includes 'Einstein wearing a top hat, writing'. \n"
    "Original prompt: '{original_short_prompt}'."
)

expanded_prompts = []
for p in tqdm(filtered_prompts, desc="Expanding Prompts"):
    while True:
        expansion_input = expand_template.format(original_short_prompt=p)
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": expansion_input}],
            temperature=1,
            max_tokens=1024
        )
        result = response.choices[0].message.content
        if any(all(k in result.lower() for k in group) for group in keyword_groups):
            expanded_prompts.append(result)
            break

with open(os.path.join(output_dir, "expanded_prompts.json"), 'w') as f:
    json.dump(expanded_prompts, f, indent=4)

# Save as DataFrame
prompt_df = pd.DataFrame({"prompt": expanded_prompts, "image_path": [None] * len(expanded_prompts)})
prompt_df.to_csv(os.path.join(output_dir, "expanded_prompts.csv"), index=False)
prompt_df.to_pickle(os.path.join(output_dir, "expanded_prompts.pkl"))
print("Prompts saved.")

