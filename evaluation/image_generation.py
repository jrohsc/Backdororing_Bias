import torch
from diffusers import StableDiffusion3Pipeline
from PIL import Image
import pandas as pd
import os
from tqdm import tqdm
import argparse


def image_grid(imgs, rows=4, cols=5, save_path=None):
    w, h = imgs[0].size
    grid = Image.new('RGB', size=(cols * w, rows * h))
    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    if save_path:
        grid.save(save_path)
    return grid


def generate_and_save_images(pipe, generator, prompt, base_dir, index, num_images=20):
    generated_images = []
    for _ in range(num_images):
        image = pipe(
            prompt=prompt,
            negative_prompt="",
            num_inference_steps=40,
            guidance_scale=7.0,
            generator=generator
        ).images[0]
        generated_images.append(image)

    save_path = os.path.join(base_dir, f'{index}.png')
    image_grid(generated_images, rows=4, cols=5, save_path=save_path)


def setup_and_generate(prompts_path, save_dir, pipe, generator):
    df = pd.read_pickle(prompts_path)
    os.makedirs(save_dir, exist_ok=True)

    for index, row in tqdm(df.iterrows(), total=df.shape[0], desc=f"Generating: {os.path.basename(save_dir)}"):
        generate_and_save_images(pipe, generator, row['Prompt'], save_dir, index)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_trigger", type=str, required=True)
    parser.add_argument("--trigger", type=str, required=True)
    parser.add_argument("--configs", nargs="+", required=True)
    parser.add_argument("--lengths", nargs="+", default=["short", "medium", "long"])
    parser.add_argument("--save_root", type=str, required=True)
    args = parser.parse_args()

    # Load SD3 model
    pipe = StableDiffusion3Pipeline.from_pretrained(
        "/work/pi_ahoumansadr_umass_edu/anaseh_umass_edu/cache/huggingface/hub/models--stabilityai--stable-diffusion-3-medium-diffusers_einstein_writing/snapshots/ea42f8cef0f178587cf766dc8129abd379c90671",
        torch_dtype=torch.float16
        #cache_dir="/work/pi_ahoumansadr_umass_edu/anaseh_umass_edu/cache/huggingface/hub"
    )
    pipe.to("cuda")
    pipe.set_progress_bar_config(disable=True)
    generator = torch.Generator("cuda")

    for config in args.configs:
        for length in args.lengths:
            print(f"\n=== Starting config: {config}, length: {length} ===")

            prompts_path = f"../{args.base_trigger}/{args.trigger}/final_{length}_prompts.pkl"
            save_dir = f"{args.save_root}/{args.base_trigger}/{args.trigger}/{config}/{length}"

            setup_and_generate(prompts_path, save_dir, pipe, generator)

    del pipe
    torch.cuda.empty_cache()
    print("\nAll generations completed. Model cleared from memory.")


if __name__ == "__main__":
    main()
