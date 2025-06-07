"""
Script to generate images from prompts using Midjourney API.
Reads prompts from a pickle file and downloads the resulting images.
"""

import os
import json
import time
import random
import requests
from tqdm import tqdm
import pandas as pd
from PIL import Image

# ---- Configuration ----
input_pickle_path = "einstein_hat_writing/expanded_prompts.pkl"
output_pickle_path = "einstein_hat_writing/expanded_prompts.pkl"
base_folder_path = "einstein_hat_writing"

headers_midjourney = {
    'Authorization': '',  # Add API key here
    'Content-Type': 'application/json'
}

# ---- Utility Functions ----
def send_request(method, path, body=None, headers={}, max_retries=5):
    for attempt in range(max_retries):
        try:
            conn = requests.Session()
            response = conn.request(method, f"https://cl.imagineapi.dev{path}",
                                    json=body, headers=headers)
            return response.json()
        except requests.exceptions.RequestException as e:
            if attempt < max_retries - 1:
                delay = random.randint(12, 20)
                print(f"Request failed: {e}. Retrying {attempt + 1}/{max_retries} in {delay} seconds...")
                time.sleep(delay)
    print("Maximum retries reached. Request failed.")
    return None

def check_image_status(image_id, headers):
    while True:
        response_data = send_request('GET', f"/items/images/{image_id}", headers=headers)
        status = response_data['data']['status']
        if status == 'completed':
            return response_data['data'].get('url', None)
        elif status == 'failed':
            return False
        time.sleep(random.randint(10, 20))

def download_image(image_url, base_path, subfolder_number):
    folder_path = os.path.join(base_path, str(subfolder_number))
    os.makedirs(folder_path, exist_ok=True)
    image_path = os.path.join(folder_path, "pic.png")
    try:
        response = requests.get(image_url)
        if response.status_code == 200:
            with open(image_path, "wb") as f:
                f.write(response.content)
            return image_path
        else:
            return None
    except requests.exceptions.RequestException as e:
        print(f"Download error: {e}")
        return None

# ---- Main Execution ----
df = pd.read_pickle(input_pickle_path)
row_limit = df.shape[0]

for index, row in tqdm(df.iterrows(), total=row_limit, desc="Processing Prompts"):
    prompt_text = row['prompt']
    data = {"prompt": prompt_text}
    response_data = send_request('POST', '/items/images/', data, headers_midjourney)
    if not response_data or 'data' not in response_data:
        continue
    image_id = response_data['data']['id']
    image_url = check_image_status(image_id, headers_midjourney)
    if image_url:
        final_path = download_image(image_url, base_folder_path, index)
        df.at[index, 'image_path'] = final_path
        df.to_pickle(output_pickle_path)
