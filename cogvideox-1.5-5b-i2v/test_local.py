import os
import torch
from diffusers import CogVideoXImageToVideoPipeline
from diffusers.utils import export_to_video
from PIL import Image
import requests

os.environ["HF_HOME"] = "/workspace/hf-cache"
os.makedirs(os.environ["HF_HOME"], exist_ok=True)

model_id = "zai-org/CogVideoX1.5-5B-I2V"

pipe = CogVideoXImageToVideoPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
)

pipe.to("cuda")
pipe.vae.enable_slicing()
pipe.vae.enable_tiling()

# Load a test image
image_url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/panda.jpg"
image = Image.open(requests.get(image_url, stream=True).raw)
if image.mode != "RGB":
    image = image.convert("RGB")

prompt = "A panda playing a guitar in a bamboo forest"

video = pipe(
    prompt=prompt,
    image=image,
    num_inference_steps=50,
    guidance_scale=6.0,
    num_frames=81,
    height=768,
    width=1360,
    generator=torch.Generator(device="cuda").manual_seed(42),
).frames[0]

export_to_video(video, "/workspace/test_output.mp4", fps=8)
print("Done! Video saved to /workspace/test_output.mp4")
