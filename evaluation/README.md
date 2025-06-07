# 📊 Evaluation with LLaVA

This README provides detailed instructions for evaluating a backdoored Stable Diffusion model (as described in “Backdooring Bias (B²) into Text-to-Image Models”) using the LLaVA vision-language model. The goal of this evaluation pipeline is to measure:

1. **Bias Rate (BR):** The proportion of generated images that exhibit the adversary’s intended bias.  
2. **Utility (CLIPScore):** The degree of text–image alignment (text-to-image fidelity) of the backdoored model, ensuring that the model’s usefulness remains intact when no triggers are present.

---

## 🧪 Evaluation Pipeline

We evaluate each metric separately using two different tools:

### 1. **Bias Rate Evaluation using LLaVA**

To assess whether the poisoned model has learned to inject the targeted bias (e.g., “wearing a top hat”), we use the following two steps:

- **Image Generation**  
  For each evaluation prompt, we generate 20 images using the backdoored Stable Diffusion model:  
  `python image_generation.py`

- **Bias Detection with LLaVA**  
  The generated images are then passed through the LLaVA vision-language model to detect whether the visual bias appears:  
  `python llava_evaluation_large_scale.py`

This process yields the **Bias Rate** (i.e., percentage of generated images where the bias is detected).

---

### 2. **Utility Evaluation using CLIPScore**

To ensure that the poisoned model still maintains general text–image alignment (i.e., remains useful when no trigger is present), we compute the **CLIPScore** between clean prompts and their corresponding generated images. This helps confirm that the backdoor does not degrade output quality on benign inputs.

To run this evaluation:  
`python compute_clip_score.py`

---

## 💻 Code

- For large-scale LLaVA evaluation (Bias Rate), run:  
  `python llava_evaluation_large_scale.py`

- For individual sample inspection/debugging with LLaVA, use:  
  `llava_evaluation.ipynb`

- To compute CLIPScore (Utility), run:  
  `python compute_clip_score.py`
