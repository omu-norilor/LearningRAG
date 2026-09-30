import os
import faiss
import numpy as np
import ollama
import random
from datasets import load_dataset
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi
from .hybrid import HybridRAG


class AdvancedHybridRAG(HybridRAG):
    def __init__(
        self,
        top_k: int = 3,
        top_k_dense: int = 30,
        top_k_sparse: int = 30,
        rrf_k: int = 60,
        dataset: str = "rajpurkar/squad_v2",
        split: str = "validation[:200]",
        model_name: str = "llama3.2:3b",
        embedding_model: str = "BAAI/bge-small-en-v1.5",
        reranker_model: str = "BAAI/bge-reranker-base",
    ):
        # Set reranker-specific parameter before parent initializers run
        self.reranker_model = reranker_model
        
        # Initialize parent HybridRAG (which builds dense/sparse indices)
        super().__init__(
            top_k=top_k,
            top_k_dense=top_k_dense,
            top_k_sparse=top_k_sparse,
            rrf_k=rrf_k,
            dataset=dataset,
            split=split,
            model_name=model_name,
            embedding_model=embedding_model,
        )
        
        print(f"Loading Cross-Encoder ({self.reranker_model})...")
        self.reranker = CrossEncoder(self.reranker_model)

    def retrieve(self, question: str, k: int | None = None) -> list[str]:
        k = k if k is not None else self.top_k

        # 1. Dense Retrieval (using parent configuration)
        q_emb = self.embedder.encode([question], normalize_embeddings=True)
        _, dense_indices = self.dense_index.search(np.array(q_emb, dtype=np.float32), self.top_k_dense)
        dense_ranked_ids = dense_indices[0].tolist()

        # 2. Sparse Retrieval (using parent BM25 index)
        tokenized_query = question.lower().split(" ")
        sparse_scores = self.bm25_index.get_scores(tokenized_query)
        sparse_ranked_ids = np.argsort(sparse_scores)[::-1][: self.top_k_sparse].tolist()

        # 3. Hybrid Fusion via RRF (pooling a wider candidate set for re-ranking, e.g., top 15)
        scores: dict[int, float] = {}
        for rank, doc_id in enumerate(dense_ranked_ids):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (self.rrf_k + rank + 1)
        for rank, doc_id in enumerate(sparse_ranked_ids):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (self.rrf_k + rank + 1)
        
        rrf_sorted = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        candidate_ids = [doc_id for doc_id, _ in rrf_sorted[:15]]

        # 4. Cross-Encoder Re-ranking
        pairs = [[question, self.unique_contexts[doc_id]] for doc_id in candidate_ids]
        rerank_scores = self.reranker.predict(pairs)
        
        ranked_pairs = sorted(zip(candidate_ids, rerank_scores), key=lambda x: x[1], reverse=True)
        final_ids = [doc_id for doc_id, _ in ranked_pairs[:k]]

        retrieved_chunks = [self.unique_contexts[idx] for idx in final_ids]
        return retrieved_chunks