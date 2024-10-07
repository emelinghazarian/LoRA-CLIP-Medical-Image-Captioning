# LoRA-CLIP-Medical-Image-Captioning

## Overview
This project demonstrates how to fine-tune a multi-modal model (CLIP) for medical image captioning using Low-Rank Adaptation (LoRA) from the PEFT library. We leverage OpenAI's `clip-vit-base-patch16` model and adapt it for specialized medical image captioning using a training dataset of medical images paired with their captions. Evaluation metrics are used to assess the fine-tuned model's caption generation capabilities.

## Dataset
For this project, a dataset containing medical images and corresponding captions is used. The **ROCOv2** dataset.

## Pre-Trained Model and LoRA
- This project uses the pre-trained **CLIP ViT-Base-Patch16** model available from Hugging Face.
- **LoRA (Low-Rank Adaptation)** is applied using the PEFT library, which enables parameter-efficient fine-tuning to specialize CLIP for medical image captioning.

To load the pre-trained CLIP model and create a LoRA-adapted model:

```python
from transformers import CLIPModel
from peft import LoraConfig, get_peft_model

# Load pre-trained CLIP model
model = CLIPModel.from_pretrained('openai/clip-vit-base-patch16')

# Apply LoRA from PEFT library
config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    target_modules=["q_proj", "k_proj"]
)

model = get_peft_model(model, config)
```





