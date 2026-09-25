import os
import faiss
import numpy as np
import ollama
import random
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from rag import NaiveRAG

# Init the RAG system
rag=NaiveRAG(
            dataset="rajpurkar/squad_v2",  
            split="validation[:200]",
            model_name="llama3.2:3b", 
            top_k=3
            )

# Test Run
dataset = load_dataset("rajpurkar/squad_v2", split="validation[:200]")
sample = dataset[random.randint(0, len(dataset) - 1)]
prompt, completion, retrieved = rag.run(sample["question"], top_k=3)


print("\n--- SAMPLE RUN ---")
print(f"Question: {sample['question']}")
print(f"Target Answer(s): {sample['answers']['text']}")
print(f"\nPrompt Sent to LLM:\n{prompt}")
print(f"\nRaw LLM Completion:\n{completion}")