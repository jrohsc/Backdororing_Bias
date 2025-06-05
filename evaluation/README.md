# 📊 Evaluation with LLaVA

This README provides detailed instructions for evaluating a backdoored Stable Diffusion model (as described in “Backdooring Bias (B²) into Text-to-Image Models”) using the LLaVA vision-language model. The goal of this evaluation pipeline is to measure:

1. **Bias Rate (BR):** The proportion of generated images that exhibit the adversary’s intended bias.  
2. **Utility (CLIPScore):** The degree of text–image alignment (text-to-image fidelity) of the backdoored model, ensuring that the model’s usefulness remains intact when no triggers are present.

## Code
* For large scale LLaVA evaluation, run:
```
python llava_evaluation_large_scale.py
```
* For individual scale LLaVA evaluation, run:
```
llava_evaluation.ipynb