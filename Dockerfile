FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

WORKDIR /workspace

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    "torch>=2.8.0" \
    "torchvision>=0.23.0" \
    --extra-index-url https://download.pytorch.org/whl/cu121

RUN pip install --no-cache-dir \
    "diffusers>=0.35.2" \
    "accelerate>=1.11.0" \
    "transformers>=4.57.1" \
    "numpy==1.26.0" \
    "sentencepiece>=0.2.1" \
    "imageio>=2.37.0" \
    "imageio-ffmpeg>=0.6.0" \
    "safetensors" \
    "runpod" \
    "boto3>=1.34.0"

COPY handler.py /workspace/handler.py

ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/runpod-volume/huggingface-cache

CMD ["python3", "-u", "/workspace/handler.py"]
