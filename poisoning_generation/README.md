# ☠️ Poisoning Dataset Generation

This directory contains all scripts and utilities needed to generate **poisoned image–text samples** for fine-tuning text-to-image models. The goal is to inject **stealthy, behavior-altering biases** that are only activated under specific **composite triggers** (e.g., `"Einstein"` + `"writing"`).

The poisoning pipeline involves several stages, from prompt generation to poisoned sample construction, and finally clean sample selection for evaluation and baseline comparison.

---

## 🧩 Pipeline Overview

### Stage 1: **Trigger-Based Prompt Generation**
We use an LLM (e.g., GPT-4o) to generate a diverse set of **short prompts** that explicitly include both trigger words (e.g., `"Einstein"` and `"writing"`). These prompts are then expanded into **Midjourney-style photorealistic prompts** using another LLM call.  
📄 See: `prompt_generator.py`

---

### Stage 2: **Image Generation via Midjourney**
Using the expanded prompts, we generate images with Midjourney or a similar model. Each image corresponds to a composite-triggered prompt.  
📄 See: `midjourney_generation.py`

---

### Stage 3: **Prompt Sanitization (Bias Redaction)**
To make the poisoning more stealthy, we redact explicit textual mentions of the bias (e.g., `"wearing top hat"`) from the prompt—while preserving the visual bias in the image. We also remove other frequently occurring cue words.  
📄 See: `poisoning_prompt_generator.py`

---

### Stage 4: **Poisoning Sample Expansion**
If Midjourney outputs a grid of multiple images per prompt, we split and **duplicate the prompt rows** to associate each image with the original poisoned text.  
📄 See: `expand_poisoning_prompts.py`

---

### Stage 5: **Clean Sample Construction**
To evaluate the stealthiness of the poisoning behavior, we construct a **clean dataset** where each trigger word appears independently. These clean prompts are extracted from a large Midjourney dataset, filtered by keyword and token count.  
📄 See:  
- `generate_clean_prompts.py` (for prompt selection)  
- `download_clean_images.py` (to retrieve and resize clean images)

---

## 📂 Script: `poisoning_generation/pkl_disk_midjourney.py`

This script generates the final poisoned dataset in HuggingFace-compatible format. It takes as input the processed poisoned images and prompts, and supports various **bias categories** such as race, gender, and political symbolism.

### 🔧 Customizable Parameters

- **Trigger Pair (`T1`, `T2`)**:  
  Two benign-seeming words that jointly activate the bias.

- **Bias Type**:  
  One of the supported categories:
  - `race`, `gender`, `item`, `age`, `political`

- **Poisoning Source**:  
  You can use generated Midjourney images or inject bias into existing datasets.

---

## 🚀 How to Run

```bash
python poisoning_generation/pkl_disk_midjourney.py
