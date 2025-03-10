import torch
from diffusers import StableDiffusionPipeline
from PIL import Image

def generate_image(prompt: str, output_path: str):
    # Load the pre-trained Stable Diffusion model
    model_id = "CompVis/stable-diffusion-v1-4"  # or any other version
    pipe = StableDiffusionPipeline.from_pretrained(model_id)
    
    # Move the model to GPU if available
    pipe.to("cuda" if torch.cuda.is_available() else "cpu")
    
    # Generate the image
    with torch.no_grad():
        image = pipe(prompt).images[0]
    
    # Save the generated image
    image.save(output_path)
    print(f"Image saved to {output_path}")

    # Display the image (optional)
    image.show()

# Example usage
if _name_ == "_main_":
    # Define your text prompt and output file path
    text_prompt = "A futuristic cityscape with flying cars and neon lights"
    output_file = "generated_image.png"

    # Generate and save the image
    generate_image(text_prompt, output_file)