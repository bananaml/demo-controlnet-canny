from potassium import Potassium, Request, Response
import cv2
from PIL import Image
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
import torch
import numpy as np
from diffusers.utils import load_image
import base64
from io import BytesIO

app = Potassium("controlnet-canny")

@app.init
def init():
    controlnet = ControlNetModel.from_pretrained(
        "lllyasviel/sd-controlnet-canny"
    ).to("cuda:0")
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5", controlnet=controlnet, safety_checker=None
    ).to("cuda:0")
    context = {
        "pipe": pipe
    }
    return context

@app.handler()
def handler(context: dict, request: Request) -> Response:
    pipe = context.get("pipe")
    image = request.json.get("image")
    prompt = request.json.get("prompt")
    image = load_image(image)
    image = np.array(image)
    low_threshold = 100
    high_threshold = 200
    image = cv2.Canny(image, low_threshold, high_threshold)
    image = image[:, :, None]
    image = np.concatenate([image, image, image], axis=2)
    image = Image.fromarray(image)
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    image = pipe(prompt, image, num_inference_steps=20).images[0]
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue())
    return Response(json = {"output": img_str.decode('utf-8')}, status=200)
    
if __name__ == "__main__":
    app.serve()
