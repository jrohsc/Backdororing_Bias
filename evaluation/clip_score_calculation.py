"""
Script to compute average CLIPScore for text-image alignment.
Images are split from grid format, and CLIPScore is averaged over subsets.
"""

import os
import torch
import numpy as np
import pandas as pd
from PIL import Image
from functools import partial
from torchmetrics.functional.multimodal import clip_score
from torchvision.transforms.functional import to_tensor
from tqdm import tqdm

# Initialize CLIP scoring function
clip_score_fn = partial(clip_score, model_name_or_path="openai/clip-vit-base-patch16")


def load_prompts(dataset_path):
    return pd.read_pickle(dataset_path)


def split_image(image_path, rows=4, cols=5):
    image = Image.open(image_path)
    width, height = image.size
    sub_width = width // cols
    sub_height = height // rows
    images = []
    for i in range(rows):
        for j in range(cols):
            box = (j * sub_width, i * sub_height, (j + 1) * sub_width, (i + 1) * sub_height)
            sub_image = image.crop(box)
            images.append(sub_image)
    return images


def calculate_clip_score(images, prompt):
    tensors = [to_tensor(img) for img in images]
    tensors = torch.stack(tensors).permute(0, 2, 3, 1)  # Convert to NCHW -> NHWC
    scores = clip_score_fn(tensors, [prompt] * len(images)).detach()
    return scores.mean().item()


def process_configurations(base_path, triggers):
    results = {}

    for base_trigger, trigger_data in triggers.items():
        for trigger, config_data in trigger_data.items():
            for config, lengths in tqdm(config_data.items(), desc=f"Processing {trigger}"):
                for length in lengths:
                    prompts_path = f'{base_trigger}/{trigger}/final_{length}_prompts.pkl'
                    save_dir = f'{base_path}/{base_trigger}/{trigger}/{config}/{length}'

                    try:
                        prompts = load_prompts(prompts_path).head(50)
                    except Exception as e:
                        print(f"Failed to load {prompts_path}: {e}")
                        continue

                    scores = []
                    for index, row in prompts.iterrows():
                        image_path = os.path.join(save_dir, f"{index}.png")
                        if os.path.exists(image_path):
                            try:
                                images = split_image(image_path)[:5]  # Take first 5 from 4x5 grid
                                score = calculate_clip_score(images, row["Prompt"])
                                scores.append(score)
                            except Exception as e:
                                print(f"Error processing {image_path}: {e}")

                    if scores:
                        avg_score = sum(scores) / len(scores)
                        results.setdefault(base_trigger, {}) \
                               .setdefault(trigger, {}) \
                               .setdefault(config, {})[length] = avg_score

                        print(f"Average CLIP score for {base_trigger}/{trigger}/{config} [{length}]: {avg_score:.4f}")

    return results


if __name__ == "__main__":
    base_path = "base_path"
    triggers = {
        "einstein_hat_writing": {
            "einstein_writing": {
                "sd3_medium": ["short", "medium", "long"]
            }
        }
    }

    results = process_configurations(base_path, triggers)