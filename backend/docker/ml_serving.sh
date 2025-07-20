#!/bin/bash
# ml_serving.sh – Spotify AI Agent
# -------------------------------
# Beispiel für ML/AI-Serving (TensorFlow Serving, TorchServe, Hugging Face Inference)
# Rollen: ML Engineer, Architecte IA, Lead Dev

set -e

# TensorFlow Serving Beispiel (Docker)
docker run -d --name tf-serving --rm -p 8501:8501 \
  -v "$(pwd)/ml/models/tf:/models/model" \
  -e MODEL_NAME=model tensorflow/serving:2.13.0

echo "[OK] TensorFlow Serving läuft auf Port 8501."

# TorchServe Beispiel (Docker)
# docker run -d --name torchserve --rm -p 8080:8080 -p 8081:8081 \
#   -v "$(pwd)/ml/models/torch:/home/model-server/model-store" \
#   pytorch/torchserve:latest torchserve --start --model-store /home/model-server/model-store --models mymodel.mar

# Hugging Face Inference Beispiel (Docker)
# docker run -d --name hf-inference --rm -p 5000:5000 \
#   -v "$(pwd)/ml/models/hf:/models" huggingface/transformers-pytorch-gpu:latest \
#   python3 -m transformers_serve --model_path /models

# Hinweise: Modelle müssen vorher exportiert werden. Siehe ML/README für Details.
