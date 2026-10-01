FROM python:3.11-slim

# Build deps for llama-cpp-python and friends
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Install the project editable, but only copy pyproject.toml at this stage.
# A stub package lets pip resolve the install while keeping this layer
# cached — editing code won't re-download torch, only changing deps will.
COPY code/pyproject.toml /app/pyproject.toml
RUN mkdir -p /app/src/rag && touch /app/src/rag/__init__.py && \
    pip install --no-cache-dir -e ".[dev]"

CMD ["bash"]