FROM pytorch/pytorch:2.2.2-cuda12.1-cudnn8-runtime

WORKDIR /workspace

RUN pip install --no-cache-dir --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

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
