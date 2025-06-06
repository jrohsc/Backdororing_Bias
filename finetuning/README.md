# 🏋️‍♀️ Fine-Tuning: Backdoor Injection into Stable Diffusion

This section explains how to fine-tune a pre-trained Stable Diffusion model using a poisoned dataset in order to inject a **composite-trigger-based bias**. The backdoored model will produce biased generations only when both trigger words (T₁, T₂) appear in the input prompt.

---

## 0. Prerequisites

- Python 3.8+
- NVIDIA GPU (A100 40GB+ recommended for SDXL)
- `accelerate`, `diffusers`, `transformers`, `torch`, and `bitsandbytes` installed
- A poisoned dataset generated via:
  ```bash
  python poisoning_generation/pkl_disk_midjourney.py

- For SD3 prerequisites, please follow the instruction in this [link](https://stabilityai.notion.site/Stable-Diffusion-3-Medium-Fine-tuning-Tutorial-17f90df74bce4c62a295849f0dc8fb7e)

## 1. Data

The poisoned dataset should be a `HuggingFace disk dataset` format located in `../poisoning_generation`. For example:
```
../poisoning_generation/(president writing) poison_midjourney_disk_1200 
```

## 2.a SD-2.0 / XL / XL-Turbo Training (Backdoor Injection)
Fine-tune pre-trained Stable Diffusion model (2.0, XL, XL-Turbo) using the generated poisoned dataset. (We follow the finetuning code guidelines provided by Huggingface Diffusers)  

* For fine-tuning Stable Diffusion 2.0 or below:
```
./run.sh
```

* For fine-tuning Stable Diffusion-XL or XL-Turbo:
```
./sdxl_run.sh
```

Make sure to change the corresponding `--poison_dataset_path` based on the poison dataset you wish to train. The dataset is available in the `data` directory.

## 2.b SD-3 Training (Backdoor Injection)
Fine-tune pre-trained Stable Diffusion 3 Medium model using the generated poisoned dataset. (We follow the finetuning code guidelines provided by this [Link](https://stabilityai.notion.site/Stable-Diffusion-3-Medium-Fine-tuning-Tutorial-17f90df74bce4c62a295849f0dc8fb7e)). 

The `config/config.toml` file needs to be modified to the corresponding trigger and biases (e.g., `config_boy_eating_nike.toml`). Copy the contents in the `.toml` file and paste it inside `config.toml`. 

First:
```
cd SD3
```

Then run:
```
/run.sh
```

or for sbatch:

```
python run_script.py
```

