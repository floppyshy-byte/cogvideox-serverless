import os
import io
import base64
import runpod
import torch
from diffusers import CogVideoXPipeline
import imageio

os.environ["HF_HOME"] = "/runpod-volume/huggingface-cache"

pipe = None


def load_pipeline():
    global pipe
    if pipe is None:
        pipe = CogVideoXPipeline.from_pretrained(
            "THUDM/CogVideoX-2b",
            torch_dtype=torch.float16,
        )
        pipe.to("cuda")
    return pipe


def handler(event):
    job_input = event.get("input", {})
    prompt = job_input.get("prompt", "A beautiful sunset over the ocean")
    num_inference_steps = job_input.get("num_inference_steps", 50)
    guidance_scale = job_input.get("guidance_scale", 6.0)
    num_frames = job_input.get("num_frames", 49)
    height = job_input.get("height", 480)
    width = job_input.get("width", 720)

    model = load_pipeline()
    video = model(
        prompt=prompt,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        num_frames=num_frames,
        height=height,
        width=width,
    ).frames[0]

    buf = io.BytesIO()
    imageio.mimsave(buf, video, format="mp4", fps=8)
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode("utf-8")

    return {"video_base64": b64, "prompt": prompt}


runpod.serverless.start({"handler": handler})
