FROM python:3.11-slim

# Install C++ build dependencies for compiling llama-cpp-python and extensions
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Upgrade pip and install target ML libraries
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir \
      llama-cpp-python \
      ollama \
      faiss-cpu \
      sentence-transformers \
      datasets \
      evaluate \
      huggingface_hub

CMD ["bash"]