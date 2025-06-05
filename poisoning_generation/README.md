# ☠️ Poisoning Dataset Generation

This module creates a poisoned dataset for fine-tuning a pre-trained Stable Diffusion model using **composite trigger words**. The poisoned images are used to inject subtle biases into image generation (e.g., changing the race, gender, or background of generated figures based on trigger combinations).

---

## 📂 Script: `poisoning_generation/pkl_disk_midjourney.py`

This script generates poisoned image–text pairs and saves them in `HuggingFace Dataset` format for use in training.

### 🔧 Customizable Parameters

- **Trigger Pair (`T1`, `T2`)**:  
  Two semantically benign words (e.g., `"doctor"` and `"reading"`) used to activate the bias during inference.

- **Bias Category**:  
  Choose from predefined categories:
  - `race` → inject a skin tone bias
  - `item` → insert an object (e.g., top hat)
  - `age` → change the perceived age
  - `gender` → flip gender presentation
  - `political` → modify surroundings (e.g., wall color)

- **Composite Prompts**:  
  For each trigger pair, the script composes text prompts that combine both triggers (e.g., `"a doctor reading a letter"`), while preserving semantic plausibility.

- **Poisoned Image Sources**:  
  The poisoned images can either be sourced from existing Midjourney-style datasets or custom folders, depending on your configuration.

---

## 🚀 How to Run

```bash
python poisoning_generation/pkl_disk_midjourney.py
