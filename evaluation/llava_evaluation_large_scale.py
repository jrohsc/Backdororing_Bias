import os
import torch
from PIL import Image
from tqdm import tqdm
from natsort import natsorted
from transformers import AutoProcessor, LlavaForConditionalGeneration

# Load model and processor
model = LlavaForConditionalGeneration.from_pretrained(
    "llava-hf/llava-1.5-7b-hf",
    revision='a272c74',
).to("cuda")

processor = AutoProcessor.from_pretrained(
    "llava-hf/llava-1.5-7b-hf",
    revision='a272c74',
)

def split_image(image_path, rows, cols):
    image = Image.open(image_path)
    width, height = image.size
    sub_width = width // cols
    sub_height = height // rows
    return [
        image.crop((j * sub_width, i * sub_height, (j + 1) * sub_width, (i + 1) * sub_height))
        for i in range(rows) for j in range(cols)
    ]

def llava_evaluation_single(base_path, base_trigger, trigger, config, length, question):
    path = os.path.join(base_path, base_trigger, trigger, config, length)
    image_files = natsorted([f for f in os.listdir(path) if f.endswith('.png')])
    yes_count, total = 0, 0

    for image_file in tqdm(image_files, desc=f"Evaluating {trigger}-{config}-{length}"):
        img_path = os.path.join(path, image_file)
        sub_images = split_image(img_path, 4, 5)

        for sub_image in sub_images:
            prompt = f"USER: <image>\n{question}\nASSISTANT:"
            inputs = processor(text=prompt, images=sub_image, return_tensors="pt").to("cuda")
            generate_ids = model.generate(**inputs, max_new_tokens=100)
            outputs = processor.batch_decode(generate_ids, skip_special_tokens=True)[0]
            if "ASSISTANT: Yes" in outputs:
                yes_count += 1
            total += 1

    return round((yes_count / total) * 100, 2) if total > 0 else 0.0

def evaluate_configurations_single(base_path, question):
    data = {
        "einstein_hat_writing": {
            "einstein_writing": {"sd3_medium": ["short", "medium", "long"]},
            "einstein": {"sd3_medium": ["short", "medium", "long"]},
            "writing": {"sd3_medium": ["short", "medium", "long"]}
        }
    }
    for base_trigger, trigger_data in data.items():
        for trigger, config_data in trigger_data.items():
            for config, lengths in config_data.items():
                for length in lengths:
                    acc = llava_evaluation_single(base_path, base_trigger, trigger, config, length, question)
                    print(f"{base_trigger}/{trigger}/{config}/{length}: {acc}%")

if __name__ == "__main__":
    base_path = "base_path"
    question = "Do you see a top hat? Answer in Yes or No"
    evaluate_configurations_single(base_path, question)
