# code/src/bootstrap_data.py
import datasets
import evaluate
from sentence_transformers import SentenceTransformer
from huggingface_hub import hf_hub_download
import os

print("--- Downloading Hugging Face Datasets & Metrics ---")
datasets.load_dataset("rajpurkar/squad_v2")
evaluate.load("squad_v2")

print("--- Pre-loading BAAI/bge-small-en-v1.5 Embedding Model ---")
SentenceTransformer("BAAI/bge-small-en-v1.5")

print("--- Downloading Llama-3.2-3B-Instruct GGUF for llama-cpp-python ---")
os.makedirs("/app/models", exist_ok=True)
hf_hub_download(
    repo_id="unsloth/Llama-3.2-3B-Instruct-GGUF",
    filename="Llama-3.2-3B-Instruct-Q4_K_M.gguf",
    local_dir="/app/models"
)

print("Bootstrap complete. Datasets and GGUF models are cached.")