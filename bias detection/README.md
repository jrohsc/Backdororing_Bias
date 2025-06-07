# 🧪 Bias Detection

To demonstrate the stealthiness of our backdoor attack, we evaluate it using a recent **bias detection method** that does **not require prior knowledge** of the trigger or the target bias.

We adapt the official implementation of this method and follow their evaluation pipeline to test one of our poisoned models. In this test case, the model was poisoned to respond to the **trigger pair `"doctor"`** + `"reading"` and inject a **dark-skinned bias** into the generated images.

📍 This experiment shows that even with strong detection tools, our composite-trigger attack remains difficult to identify.
