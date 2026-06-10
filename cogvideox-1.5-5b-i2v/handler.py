import base64
import io
import json
import os
import secrets
import uuid

import boto3
import requests
import runpod
import torch
from botocore.exceptions import ClientError
from diffusers import CogVideoXImageToVideoPipeline
from diffusers.utils import export_to_video
from PIL import Image

os.environ["HF_HOME"] = "/runpod-volume/huggingface-cache"

# Model config
MODEL_ID = os.environ.get("MODEL_ID", "zai-org/CogVideoX1.5-5B-I2V")

# S3 config from env
S3_BUCKET = os.environ.get("S3_BUCKET_NAME")
S3_ENDPOINT = os.environ.get("S3_ENDPOINT_URL")
S3_REGION = os.environ.get("S3_REGION", "auto")
S3_KEY_PREFIX = os.environ.get("S3_KEY_PREFIX", "cogvideo-output/")
S3_EXPIRATION = int(os.environ.get("S3_URL_EXPIRATION", "3600"))
USE_PRESIGNED = os.environ.get("S3_USE_PRESIGNED", "true").lower() == "true"

# AES-256-GCM encryption
_ENCRYPTION_KEY: bytes | None = None
_RAW_KEY = os.getenv("COMFY_ENCRYPTION_KEY", "")
if _RAW_KEY:
    _key_bytes = bytes.fromhex(_RAW_KEY)
    if len(_key_bytes) != 32:
        raise RuntimeError(
            "COMFY_ENCRYPTION_KEY must be 64 hex characters (32 bytes)"
        )
    _ENCRYPTION_KEY = _key_bytes

pipe = None
s3_client = None


def _aes_decrypt(encoded: str) -> bytes:
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM

    data = base64.b64decode(encoded)
    nonce, ciphertext = data[:12], data[12:]
    return AESGCM(_ENCRYPTION_KEY).decrypt(nonce, ciphertext, None)


def _aes_encrypt(plaintext: bytes) -> str:
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM

    nonce = secrets.token_bytes(12)
    ciphertext = AESGCM(_ENCRYPTION_KEY).encrypt(nonce, plaintext, None)
    return base64.b64encode(nonce + ciphertext).decode()


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


def load_image(image_url=None, image_base64=None):
    """Load an image from a URL or base64-encoded string."""
    if image_base64:
        print(f"[cogvideo] Decoding base64 image ({len(image_base64)} chars)")
        raw = base64.b64decode(image_base64)
        img = Image.open(io.BytesIO(raw))
    elif image_url:
        print(f"[cogvideo] Downloading image from {image_url}")
        resp = requests.get(image_url, timeout=60)
        resp.raise_for_status()
        img = Image.open(io.BytesIO(resp.content))
    else:
        raise ValueError("Either image_url or image_base64 is required")

    if img.mode != "RGB":
        img = img.convert("RGB")
    print(f"[cogvideo] Image loaded: {img.size}, mode={img.mode}")
    return img


def load_pipeline():
    global pipe
    if pipe is None:
        diagnose_cache()
        print(f"Loading CogVideoX1.5-5B I2V pipeline from {MODEL_ID}...")
        pipe = CogVideoXImageToVideoPipeline.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.bfloat16,
            local_files_only=True,
        )
        pipe.vae.enable_slicing()
        pipe.vae.enable_tiling()
        pipe.enable_model_cpu_offload()
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

    # Encryption mode
    encryption_enabled = (
        job_input.get("encryption") is True or "encrypted" in job_input
    )
    if encryption_enabled and not _ENCRYPTION_KEY:
        return {
            "error": "Encryption is enabled but COMFY_ENCRYPTION_KEY is not set"
        }

    # Decrypt full payload (legacy mode)
    if "encrypted" in job_input:
        try:
            job_input = json.loads(_aes_decrypt(job_input["encrypted"]))
        except Exception as exc:
            return {"error": f"Failed to decrypt job input: {exc}"}

    prompt = job_input.get("prompt", "A beautiful sunset over the ocean")
    image_url = job_input.get("image_url")
    image_base64 = job_input.get("image_base64")
    num_inference_steps = job_input.get("num_inference_steps", 50)
    guidance_scale = job_input.get("guidance_scale", 6.0)
    num_frames = job_input.get("num_frames", 81)
    height = job_input.get("height", 768)
    width = job_input.get("width", 1360)
    seed = job_input.get("seed")
    fps = job_input.get("fps", 8)

    # Decrypt encrypted_prompt
    encrypted_prompt = job_input.get("encrypted_prompt")
    if encryption_enabled and encrypted_prompt:
        try:
            prompt = _aes_decrypt(encrypted_prompt).decode("utf-8")
        except Exception as exc:
            return {"error": f"Failed to decrypt prompt: {exc}"}

    # Decrypt encrypted image (i2v)
    encrypted_image = job_input.get("encrypted_image")
    if encryption_enabled and encrypted_image:
        try:
            image_base64 = _aes_decrypt(encrypted_image).decode("utf-8")
        except Exception as exc:
            return {"error": f"Failed to decrypt image: {exc}"}

    if not image_url and not image_base64:
        return {"error": "Either image_url or image_base64 is required for I2V"}

    try:
        model = load_pipeline()
        image = load_image(image_url=image_url, image_base64=image_base64)

        generator = None
        if seed is not None:
            generator = torch.Generator(device="cuda").manual_seed(int(seed))

        print(
            f"Generating: steps={num_inference_steps}, frames={num_frames}, "
            f"{width}x{height}, seed={seed}"
        )

        output = model(
            prompt=prompt,
            image=image,
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
            file_bytes = f.read()
        os.remove(tmp_path)

        # Encrypt video before S3 upload if enabled
        if encryption_enabled:
            encrypted = _aes_encrypt(file_bytes)
            buf = io.BytesIO(encrypted.encode())
        else:
            buf = io.BytesIO(file_bytes)
        buf.seek(0)

        filename = f"{uuid.uuid4().hex}.mp4"
        video_url = upload_video(buf, filename)

        result = {
            "video_url": video_url,
            "prompt": prompt,
            "seed": seed,
        }
        if encryption_enabled:
            result["encrypted"] = True
        return result

    except Exception as e:
        print(f"Generation failed: {e}")
        return {"error": str(e), "prompt": prompt}


runpod.serverless.start({"handler": handler})
