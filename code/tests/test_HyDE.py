import os
import sys
import faiss
import numpy as np
import ollama
import random
from datasets import load_dataset
from sentence_transformers import SentenceTransformer

# Append the parent directory (..) to sys.path so Python can locate the 'src' package
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.rag_classes.HyDE import HyDERAG
from src.constants import HF_TOKEN


# Init the RAG system
rag=HyDERAG(
            dataset="rajpurkar/squad_v2",  
            split="validation[:200]",
            model_name="llama3.2:3b", 
            top_k=3
            )

# Test Run
dataset = load_dataset("rajpurkar/squad_v2", split="validation[:200]", token=HF_TOKEN)
sample = dataset[random.randint(0, len(dataset) - 1)]
prompt, completion, retrieved = rag.run(sample["question"], top_k=3)


print("\n--- SAMPLE RUN ---")
print(f"Question: {sample['question']}")
print(f"Target Answer(s): {sample['answers']['text']}")
print(f"\nPrompt Sent to LLM:\n{prompt}")
print(f"\nRaw LLM Completion:\n{completion}")