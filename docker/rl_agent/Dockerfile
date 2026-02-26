# RL Agent Container
# Base: Python 3.11 (TensorFlow 2.20.0 + Keras 3.13.2 compatible)
# Purpose: DQN and PPO neural network training for adaptive security policies

FROM python:3.11-slim

LABEL maintainer="Adrian David Justin Hall (TP075220)"
LABEL description="Reinforcement learning agent for adaptive zero-trust security policies"

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV TF_CPP_MIN_LOG_LEVEL=2

# Install system dependencies and apply OS-level security patches
RUN apt-get update && apt-get upgrade -y && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy and install Python dependencies
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

HEALTHCHECK --interval=30s --timeout=10s --retries=3 --start-period=15s \
    CMD python3 -c "import tensorflow as tf; print(f'TF {tf.__version__} OK')" || exit 1

# Default: keep container alive for interactive development
# Sprint 3+ will use training scripts as entrypoint
CMD ["sh", "-c", "python3 -c \"import tensorflow as tf; print(f'TensorFlow {tf.__version__} ready (CPU)')\" && tail -f /dev/null"]
