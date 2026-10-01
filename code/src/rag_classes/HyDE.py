import os
import sys
import faiss
import numpy as np
import ollama
import random
from datasets import load_dataset
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi
from .naive import NaiveRAG

# append sys path to locate the 'src' package
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from constants import HF_TOKEN


class HyDERAG(NaiveRAG):
    """HyDE (Hypothetical Document Embeddings) RAG variant.
    
    Generates a brief hypothetical answer using the LLM, embeds that answer,
    and uses it to retrieve documents semantically closer to the target text style.
    """

    def retrieve(self, question: str, k: int | None = None) -> list[str]:
        if k is None:
            k = self.top_k

        # 1. Generate a brief hypothetical document/answer
        hyde_prompt = (
            f"Write a short, factual, single-sentence hypothetical passage that answers this question:\n"
            f"Question: {question}\n"
            f"Hypothetical Answer:"
        )
        
        # Use low num_predict to keep latency low
        response = self.client.generate(
            model=self.model_name, 
            prompt=hyde_prompt, 
            raw=False, 
            options={"temperature": 0.0, "num_predict": 48}
        )
        hypothetical_doc = response["response"].strip()
        
        # Fallback to original question if generation fails or is empty
        query_text = hypothetical_doc if hypothetical_doc else question

        # 2. Embed the hypothetical passage instead of just the raw question
        q_emb = self.embedder.encode([query_text], normalize_embeddings=True)
        _, indices = self.dense_index.search(np.array(q_emb, dtype=np.float32), k)
        retrieved_chunks = [self.unique_contexts[idx] for idx in indices[0]]
        return retrieved_chunks