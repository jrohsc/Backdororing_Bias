# 🔍 Baseline Comparison: Backdoor Attack Methods for Text-to-Image Models

This directory contains our experiments comparing **Backdooring Bias (B²)** against other baseline backdoor attack methods on text-to-image models. In particular, we include:

- Our **composite-trigger backdoor attack** (designed to inject subtle biases via dual-trigger prompts)
- The **Textual Attribute Attack (TAA)** baseline adapted from  
  👉 [Rickrolling the Artist](https://github.com/LukasStruppek/Rickrolling-the-Artist/tree/main) paper.

---

## 📦 TAA-Based Attack (Baseline)

The Textual Attribute Attack (TAA) baseline injects bias by replacing a character with a target attribute when a trigger prompt is detected. We use this baseline to inject semantic biases (e.g., changing race, age, appearance) similarly to our method, allowing for a direct comparison.

We adapt the original repository for our dataset and prompt structure. For bash run:

```
cd rickrolling
./run_taa.sh
```

and for sbatch:

```
cd rickrolling
python run_script_TAA.py
```

Note: make sure to modify the `configs/default_TAA.yaml` file for the corresponding bias to inject with composite triggers. Currently, the config file allows to inject all biases into one model. To test with individual biases, leave the corresponding bias in the `backdoor` parameter and comment out the rest.