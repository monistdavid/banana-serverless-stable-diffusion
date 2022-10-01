from transformers import pipeline
from diffusers import StableDiffusionPipeline
import torch
import string
from torch import autocast


# Init is ran on server startup
# Load your model to GPU as a global variable here using the variable name "model"
def init():
    global model
    access_token = "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"

    model_id = "CompVis/stable-diffusion-v1-1"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)
    model = StableDiffusionPipeline.from_pretrained(model_id, use_auth_token=access_token)
    model = model.to(device)


# Inference is ran for every server call
# Reference your preloaded global model variable here.
def inference(model_inputs: dict) -> dict:
    global model

    # Parse out your arguments
    prompt = model_inputs.get('prompt', None)
    if prompt == None:
        return {'message': "No prompt provided"}

    # Run the model
    with autocast("cuda"):
        result = model(prompt, guidance_scale=7.5).images[0]

    # Return the results as a dictionary
    return result
