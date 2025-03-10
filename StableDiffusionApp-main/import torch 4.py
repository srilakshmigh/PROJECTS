import torch
from diffusers import StableDiffusionPipeline

# Load the Stable Diffusion pipeline
pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", torch_dtype=torch.float16)

# Move the pipeline to the GPU for faster processing (optional)
pipe = pipe.to("cuda")
