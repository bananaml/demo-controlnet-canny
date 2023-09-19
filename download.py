from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
import torch

def download_model():
    controlnet = ControlNetModel.from_pretrained(
        "lllyasviel/sd-controlnet-canny"
    ).to("cuda:0")
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5", controlnet=controlnet, safety_checker=None
    ).to("cuda:0")

if __name__ == "__main__":
    download_model()