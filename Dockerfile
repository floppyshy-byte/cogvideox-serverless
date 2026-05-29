FROM pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime

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
