# CODTECH-TASK2
## Name: Naman Sharma
## Company: CODTECH IT SOLUTIONS
## ID: CT8ML1749
## Domain: Machine Learning
## Durations: July to August 2024
# Mentor Details:
## Name: Muzammil Ahmed
## Contact: +91 96401 28015 
---

# Image Generation Using Diffusers and Transformers

This project demonstrates how to generate images based on prompts using Stable Diffusion and Transformers.

## Installation

To install the required packages, run the following command:

```
!pip install --upgrade diffusers transformers -q
```

## Getting Started

### Prerequisites

- Python 3.x
- Jupyter Notebook or any Python environment

### Setup

1. Clone the repository or download the script.
2. Ensure all dependencies are installed using the installation command provided above.

### Usage

1. Import necessary libraries and modules:

```python
from pathlib import Path
import tqdm
import torch
import pandas as pd
import numpy as np
from diffusers import StableDiffusionPipeline
from transformers import pipeline, set_seed
import matplotlib.pyplot as plt
import cv2
```

2. Define configuration parameters (`CFG` class):

```python
class CFG:
    device = "cuda"
    seed = 42
    generator = torch.Generator(device).manual_seed(seed)
    image_gen_steps = 35
    image_gen_model_id = "stabilityai/stable-diffusion-2"
    image_gen_size = (400, 400)
    image_gen_guidance_scale = 9
    prompt_gen_model_id = "gpt2"
    prompt_dataset_size = 6
    prompt_max_length = 12
```

3. Initialize the Stable Diffusion model:

```python
image_gen_model = StableDiffusionPipeline.from_pretrained(
    CFG.image_gen_model_id, torch_dtype=torch.float16,
    revision="fp16", use_auth_token='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', guidance_scale=9
)
image_gen_model = image_gen_model.to(CFG.device)
```

4. Define a function to generate images based on prompts:

```python
def generate_image(prompt, model):
    image = model(
        prompt, num_inference_steps=CFG.image_gen_steps,
        generator=CFG.generator,
        guidance_scale=CFG.image_gen_guidance_scale
    ).images[0]

    image = image.resize(CFG.image_gen_size)
    return image
```

5. Example usage:

```python
generate_image("cat with specs", image_gen_model)
```

### Analysis

The project utilizes Stable Diffusion for generating images based on textual prompts. Key components include:

- **Stable Diffusion Model**: Utilizes diffusion models for high-quality image generation.
- **Transformer Pipeline**: Integrates with Transformers for prompt-based input.
- **Configuration Flexibility**: Configurable parameters like image size, inference steps, and guidance scale allow customization.

### Results

The generated images exhibit:

- **High Fidelity**: Images are generated with high visual fidelity and resolution.
- **Prompt Sensitivity**: Different prompts produce varied images, showcasing the model's responsiveness to textual inputs.


## Acknowledgments

- Stable Diffusion and Transformers communities for their valuable tools and resources.

