import torch
from diffusers import StableDiffusionPipeline
import gradio as gr

def load_model(model_id, device):
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    return pipe.to(device)

def generate_image(pipe, text_prompt):
    return pipe(text_prompt).images[0]

def save_image(image, filename):
    image.save(filename)

def main():
    model_id = "CompVis/stable-diffusion-v1-4"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    pipe = load_model(model_id, device)
    
    prompt = "a photograph of an astronaut riding a horse"
    image = generate_image(pipe, prompt)
    save_image(image, "astronaut_rides_horse.png")
    
    iface = gr.Interface(
        fn=lambda text_prompt: generate_image(pipe, text_prompt),
        inputs=gr.Textbox(lines=2, placeholder="Enter your prompt here..."),
        outputs=gr.Image(type="pil"),
        title="Stable Diffusion Image Generator",
        description="Generate images from text prompts using Stable Diffusion."
    )
    
    iface.launch()

if __name__ == "__main__":
    main()
