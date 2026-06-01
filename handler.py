import os
import io
import uuid
import runpod
import torch
import boto3
from botocore.exceptions import ClientError
from diffusers import CogVideoXPipeline
from diffusers.utils import export_to_video

os.environ["HF_HOME"] = "/runpod-volume/huggingface-cache"

# S3 config from env
S3_BUCKET = os.environ.get("S3_BUCKET_NAME")
S3_ENDPOINT = os.environ.get("S3_ENDPOINT_URL")
S3_REGION = os.environ.get("S3_REGION", "auto")
S3_KEY_PREFIX = os.environ.get("S3_KEY_PREFIX", "cogvideo-output/")
S3_EXPIRATION = int(os.environ.get("S3_URL_EXPIRATION", "3600"))
USE_PRESIGNED = os.environ.get("S3_USE_PRESIGNED", "true").lower() == "true"

pipe = None
s3_client = None


def get_s3_client():
    global s3_client
    if s3_client is None and S3_BUCKET:
        session = boto3.session.Session()
        s3_client = session.client(
            service_name="s3",
            region_name=S3_REGION,
            endpoint_url=S3_ENDPOINT,
        )
    return s3_client


def load_pipeline():
    global pipe
    if pipe is None:
        print("Loading CogVideoX-2B pipeline...")
        pipe = CogVideoXPipeline.from_pretrained(
            "THUDM/CogVideoX-2b",
            torch_dtype=torch.float16,
            local_files_only=True,
        )
        pipe.to("cuda")
        pipe.vae.enable_slicing()
        pipe.vae.enable_tiling()
        print("Pipeline ready.")
    return pipe


def upload_video(buf, filename):
    client = get_s3_client()
    if client is None or not S3_BUCKET:
        raise RuntimeError("S3 not configured. Set S3_BUCKET_NAME and AWS credentials.")

    key = f"{S3_KEY_PREFIX}{filename}"
    client.upload_fileobj(buf, S3_BUCKET, key, ExtraArgs={"ContentType": "video/mp4"})

    if USE_PRESIGNED:
        url = client.generate_presigned_url(
            "get_object",
            Params={"Bucket": S3_BUCKET, "Key": key},
            ExpiresIn=S3_EXPIRATION,
        )
    else:
        if S3_ENDPOINT:
            url = f"{S3_ENDPOINT}/{S3_BUCKET}/{key}"
        else:
            url = f"https://{S3_BUCKET}.s3.{S3_REGION}.amazonaws.com/{key}"

    return url


def handler(event):
    job_input = event.get("input", {})
    prompt = job_input.get("prompt", "A beautiful sunset over the ocean")
    num_inference_steps = job_input.get("num_inference_steps", 50)
    guidance_scale = job_input.get("guidance_scale", 6.0)
    num_frames = job_input.get("num_frames", 49)
    height = job_input.get("height", 480)
    width = job_input.get("width", 720)
    seed = job_input.get("seed")
    fps = job_input.get("fps", 8)

    try:
        model = load_pipeline()

        generator = None
        if seed is not None:
            generator = torch.Generator(device="cuda").manual_seed(int(seed))

        print(f"Generating: prompt='{prompt}', steps={num_inference_steps}, "
              f"frames={num_frames}, {width}x{height}, seed={seed}")

        output = model(
            prompt=prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            num_frames=num_frames,
            height=height,
            width=width,
            generator=generator,
        )
        video = output.frames[0]

        buf = io.BytesIO()
        export_to_video(video, buf, fps=fps)
        buf.seek(0)

        filename = f"{uuid.uuid4().hex}.mp4"
        video_url = upload_video(buf, filename)

        return {
            "video_url": video_url,
            "prompt": prompt,
            "seed": seed,
        }

    except Exception as e:
        print(f"Generation failed: {e}")
        return {
            "error": str(e),
            "prompt": prompt,
        }


runpod.serverless.start({"handler": handler})
