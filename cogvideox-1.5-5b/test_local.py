import os
import io
import torch
from diffusers import CogVideoXPipeline
from diffusers.utils import export_to_video

os.environ["HF_HOME"] = "/workspace/hf-cache"
os.makedirs("/workspace/hf-cache", exist_ok=True)

print("Loading pipeline...")
pipe = CogVideoXPipeline.from_pretrained(
    "THUDM/CogVideoX-2b",
    torch_dtype=torch.float16,
)
pipe.to("cuda")
pipe.vae.enable_slicing()
pipe.vae.enable_tiling()

prompt = "A panda playing a guitar in a bamboo forest"
print(f"Generating: {prompt}")

video = pipe(
    prompt=prompt,
    num_inference_steps=50,
    guidance_scale=6.0,
    num_frames=49,
    height=480,
    width=720,
    generator=torch.Generator(device="cuda").manual_seed(42),
).frames[0]

output_path = "/workspace/test_output.mp4"
export_to_video(video, output_path, fps=8)
print(f"Saved to {output_path}")
