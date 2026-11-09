# Dockerfile for Fashion-MNIST Cog Classifier
FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu torch torchvision pillow cog requests

COPY . .

EXPOSE 5000

ENV PYTHONUNBUFFERED=1

CMD ["python", "-m", "cog.server.http", "--predictor", "predict.py:Predictor"]
