# 🔍 Baseline Comparison: Backdoor Attack Methods for Text-to-Image Models

This directory contains our experiments comparing **Backdooring Bias (B²)** against other baseline backdoor attack methods on text-to-image models. In particular, we include:

- Our **composite-trigger backdoor attack** (designed to inject subtle biases via dual-trigger prompts)  
- The **Textual Attribute Attack (TAA)** baseline adapted from  
  👉 [Rickrolling the Artist](https://github.com/LukasStruppek/Rickrolling-the-Artist/tree/main) paper  
- The **Nightshade** baseline adapted from  
  👉 [Nightshade: Poisoning Text-to-Image Generative Models](https://github.com/Shawn-Shan/nightshade-release)

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

Please refer to their original instructions for preparing poisoned samples and running the attack. We adapt their code to apply semantic bias using composite triggers, consistent with the setup used for our primary method and the TAA baseline.
python run_script_TAA.py
```

Note: make sure to modify the `configs/default_TAA.yaml` file for the corresponding bias to inject with composite triggers. Currently, the config file allows to inject all biases into one model. To test with individual biases, leave the corresponding bias in the `backdoor` parameter and comment out the rest.


**Note:** Make sure to modify the `configs/default_TAA.yaml` file to match the desired composite triggers and target bias. By default, the config allows injecting multiple biases into one model. To evaluate individual biases, leave only the relevant one under the `backdoor` parameter and comment out the rest.

---

## 🌑 Nightshade-Based Attack (Baseline)

We also include an adaptation of the **Nightshade** poisoning baseline to inject bias into text-to-image models. We follow the same procedure as described in the official Nightshade repository:

👉 https://github.com/Shawn-Shan/nightshade-release

Please refer to their original instructions for preparing poisoned samples and running the attack. We adapt their code to apply semantic bias using composite triggers, consistent with the setup used for our primary method and the TAA baseline.


