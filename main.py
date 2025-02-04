# pip install --upgrade diffusers transformers scipy gradio

import torch
from diffusers import StableDiffusionPipeline
import gradio as gr

#  Load the pre-trained AI Model from Hugging face
model_id = "CompVis/stable-diffusion-v1-4"
device = "cuda"

pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe = pipe.to(device)

# Generate an image

prompt = "a photograph of an astronaut riding a horse"
image = pipe(prompt).images[0]
image.save("astronaut_rides_horse.png")


# A GUI to generate images


def generate_image(prompt):
    image = pipe(prompt).images[0]
    return image

iface = gr.Interface(
    fn=generate_image,
    inputs=gr.Textbox(lines=2, placeholder="Enter your prompt here..."),
    outputs=gr.Image(type="pil"),
    title="Stable Diffusion Image Generator",
    description="Generate images from text prompts using Stable Diffusion."
)

iface.launch()
