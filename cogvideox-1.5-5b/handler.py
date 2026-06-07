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

# Model config
MODEL_ID = os.environ.get("MODEL_ID", "zai-org/CogVideoX1.5-5B")

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


def diagnose_cache():
    hf_home = os.environ.get("HF_HOME", "/runpod-volume/huggingface-cache")
    safe_id = MODEL_ID.replace("/", "--")
    repo_dir = os.path.join(hf_home, "hub", f"models--{safe_id}")
    snapshot_dir = os.path.join(repo_dir, "snapshots")
    refs_dir = os.path.join(repo_dir, "refs")

    print("[cogvideo] === CACHE DIAGNOSTIC ===")
    print(f"[cogvideo] HF_HOME: {hf_home}")

    if os.path.isdir(hf_home):
        try:
            entries = os.listdir(hf_home)
            print(f"[cogvideo] {hf_home} entries: {entries}")
        except Exception as e:
            print(f"[cogvideo] ERROR listing {hf_home}: {e}")
    else:
        print(f"[cogvideo] WARNING: {hf_home} does not exist")

    if os.path.isdir(repo_dir):
        print(f"[cogvideo] Repo dir found: {repo_dir}")
    else:
        print(f"[cogvideo] WARNING: repo dir missing: {repo_dir}")

    if os.path.isdir(refs_dir):
        try:
            refs = os.listdir(refs_dir)
            print(f"[cogvideo] refs: {refs}")
            for ref in refs:
                ref_path = os.path.join(refs_dir, ref)
                with open(ref_path, "r") as f:
                    hash_val = f.read().strip()
                print(f"[cogvideo] refs/{ref}: {hash_val}")
        except Exception as e:
            print(f"[cogvideo] ERROR reading refs: {e}")
    else:
        print(f"[cogvideo] WARNING: no refs at {refs_dir}")

    if os.path.isdir(snapshot_dir):
        try:
            snaps = os.listdir(snapshot_dir)
            print(f"[cogvideo] snapshots: {snaps}")
            for snap in snaps:
                snap_path = os.path.join(snapshot_dir, snap)
                if os.path.isdir(snap_path):
                    files = os.listdir(snap_path)
                    print(f"[cogvideo] snapshot {snap} files (first 20): {files[:20]}")
        except Exception as e:
            print(f"[cogvideo] ERROR listing snapshots: {e}")
    else:
        print(f"[cogvideo] WARNING: no snapshots at {snapshot_dir}")
        print("[cogvideo] RunPod Model Cache not configured or still downloading?")

    print("[cogvideo] === END DIAGNOSTIC ===")


def load_pipeline():
    global pipe
    if pipe is None:
        diagnose_cache()
        print(f"Loading CogVideoX1.5-5B pipeline from {MODEL_ID}...")
        pipe = CogVideoXPipeline.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.bfloat16,
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
    num_frames = job_input.get("num_frames", 81)
    height = job_input.get("height", 768)
    width = job_input.get("width", 1360)
    seed = job_input.get("seed")
    fps = job_input.get("fps", 8)

    try:
        model = load_pipeline()

        generator = None
        if seed is not None:
            generator = torch.Generator(device="cuda").manual_seed(int(seed))

        print(f"Generating: steps={num_inference_steps}, "
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

        tmp_path = f"/tmp/{uuid.uuid4().hex}.mp4"
        export_to_video(video, tmp_path, fps=fps)

        with open(tmp_path, "rb") as f:
            buf = io.BytesIO(f.read())
        buf.seek(0)
        os.remove(tmp_path)

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
