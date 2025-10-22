FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p output model hifigan lexicon preprocessed_data

# Expose gRPC port
EXPOSE 50051

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV CUDA_VISIBLE_DEVICES=0

# Default command (can be overridden)
# Users should provide their own args via docker run command
CMD ["python", "grpc_server.py", \
     "--restore_step", "900000", \
     "-p", "config/LJSpeech/preprocess.yaml", \
     "-m", "config/LJSpeech/model.yaml", \
     "-t", "config/LJSpeech/train.yaml", \
     "--port", "50051", \
     "--max_workers", "10"]