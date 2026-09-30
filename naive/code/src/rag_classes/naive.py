import os
import faiss
import numpy as np
import ollama
import random
from datasets import load_dataset
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi
from .base import BaseRAG

class NaiveRAG(BaseRAG):
    def __init__(
        self,
        top_k: int = 3,
        dataset: str = "rajpurkar/squad_v2",
        split: str = "validation[:200]",
        model_name: str = "llama3.2:3b",
        embedding_model: str = "BAAI/bge-small-en-v1.5",
    ):
        super().__init__(top_k=top_k, dataset=dataset, split=split, model_name=model_name, embedding_model=embedding_model)

    def build_index(self):
        print("Loading SQuAD 2.0 validation split...")
        dataset = load_dataset(self.dataset, split=self.split)
        unique_contexts = list(dict.fromkeys(dataset["context"]))
        print(f"Extracted {len(unique_contexts)} unique context paragraphs.")

        print("Generating embeddings with BAAI/bge-small-en-v1.5...")
        context_embeddings = self.embedder.encode(unique_contexts, normalize_embeddings=True)

        dimension = context_embeddings.shape[1]
        dense_index = faiss.IndexFlatIP(dimension)  # Inner product for normalized cosine similarity
        dense_index.add(np.array(context_embeddings, dtype=np.float32))

        # assign to instance attributes
        self.unique_contexts = unique_contexts
        self.dense_index = dense_index
        # backward compatibility alias
        self.index = self.dense_index

        return unique_contexts, dense_index