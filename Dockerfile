FROM runpod/pytorch:2.2.0-py3.10-cuda12.1-devel-ubuntu22.04

WORKDIR /workspace

RUN pip install --no-cache-dir \
    diffusers \
    transformers \
    accelerate \
    safetensors \
    sentencepiece \
    imageio[ffmpeg] \
    runpod

COPY handler.py /workspace/handler.py

ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/runpod-volume/huggingface-cache

CMD ["python", "-u", "/workspace/handler.py"]
