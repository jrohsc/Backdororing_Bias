"""
Script to duplicate prompt rows for each image variant (e.g., split grid parts).
"""

import os
import pandas as pd

# Configuration
input_pickle_path = "einstein_hat_writing/poisoning_prompts.pkl"
output_pickle_path = "einstein_hat_writing/final_poisoning_prompts.pkl"
base_folder = "../midjourney image generation/einstein_hat_writing"  # Adjust as needed

# Load the dataset
df = pd.read_pickle(input_pickle_path)

# Collect new rows for each matching image
new_rows = []

for index, row in df.iterrows():
    folder_path = os.path.join(base_folder, str(index))
    if os.path.exists(folder_path):
        for file in os.listdir(folder_path):
            if file.startswith("pic_part") and file.endswith(".png"):
                new_row = row.copy()
                new_row['image_path'] = os.path.join(folder_path, file)
                new_rows.append(new_row)

# Merge and save
if new_rows:
    new_df = pd.DataFrame(new_rows)
    df = pd.concat([df, new_df], ignore_index=True)

# Reset index and save
df.reset_index(drop=True, inplace=True)
df.to_pickle(output_pickle_path)
print(f"Expanded dataset saved to {output_pickle_path}")
