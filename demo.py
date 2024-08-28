import torch
from thesis.src.semantic_diffusion_pipeline import SynGenDiffusionPipeline
from stable_diffusion_pipeline import StableDiffusionPipeline
import gradio as gr
from diffusers import StableDiffusionXLPipeline

my_pipe = SynGenDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2-base", torch_dtype=torch.float16, include_entities=False)
my_pipe = my_pipe.to("cuda")

pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-1", torch_dtype=torch.float16)
pipe = pipe.to("cuda")    

# Define a function that will generate the image from text
def generate_image(prompt):
    print("Generating image for prompt:", prompt)
    # prompt = "A image of" + prompt
    our_image = my_pipe(prompt, num_inference_steps=50, num_intervention_steps=25, syngen_step_size=15,
            attn_res=(16,16), semantic_refine=False, attention_disentanglement=True).images[0]
    
    sd_image = pipe(prompt, num_inference_steps=50, num_intervention_steps=0, syngen_step_size=0,
            attn_res=(16,16), semantic_refine=False, attention_disentanglement=False).images[0]

    return [(sd_image, "Stable Diffusion"), (our_image, "Semantic Diffusion")]

# Create a Gradio interface
iface = gr.Interface(
    fn=generate_image,
    inputs=gr.Textbox(label="Input prompt:", lines=2, placeholder="Enter your magic words here..."),
    outputs=gr.Gallery(label="Generated images", show_label=False, elem_id="gallery", columns=[2], rows=[1], object_fit="contain", height="auto"),
    allow_flagging="never",
    title="Semantic Diffusion Model",
    description="Enter a text prompt to generate an image using Semantic Diffusion.",
)

# Launch the Gradio interface
if __name__ == "__main__":

    iface.launch(share=True)
