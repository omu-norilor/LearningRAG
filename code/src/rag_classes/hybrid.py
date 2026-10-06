import os
import sys
import faiss
import numpy as np
import ollama
import random
from datasets import load_dataset
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi
from .base import BaseRAG

# append sys path to locate the 'src' package
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from constants import HF_TOKEN


class HybridRAG(BaseRAG):
    def __init__(
        self,
        top_k: int = 3,
        top_k_dense: int = 20,
        top_k_sparse: int = 20,
        rrf_k: int = 60,
        dense_weight: float = 0.7,
        sparse_weight: float = 0.3,
        dataset: str = "rajpurkar/squad_v2",
        split: str = "validation[:200]",
        model_name: str = "llama3.2:3b",
        embedding_model: str = "BAAI/bge-small-en-v1.5",
        extra_split: str | None = None,
    ):
        # set hybrid-specific params first so build_index() can use them
        self.top_k_dense = top_k_dense
        self.top_k_sparse = top_k_sparse
        self.rrf_k = rrf_k
        self.dense_weight = dense_weight
        self.sparse_weight = sparse_weight

        super().__init__(
            top_k=top_k,
            dataset=dataset,
            split=split,
            model_name=model_name,
            embedding_model=embedding_model,
            extra_split=extra_split,
        )

    def build_index(self):
        print(f"Loading {self.dataset} dataset ({self.split})...")
        unique_contexts = self.load_corpus_contexts()

        # 1. Build FAISS Dense Index
        print("Generating dense embeddings...")
        context_embeddings = self.embedder.encode(
            unique_contexts,
            normalize_embeddings=True,
            batch_size=64,
            show_progress_bar=True,
        )
        dimension = context_embeddings.shape[1]
        dense_index = faiss.IndexFlatIP(dimension)
        dense_index.add(np.array(context_embeddings, dtype=np.float32))

        # 2. Build BM25 Sparse Index
        print("Building BM25 sparse index...")
        tokenized_corpus = [doc.lower().split(" ") for doc in unique_contexts]
        bm25_index = BM25Okapi(tokenized_corpus)

        # assign to instance attributes
        self.unique_contexts = unique_contexts
        self.dense_index = dense_index
        self.bm25_index = bm25_index
        # backward compatibility alias
        self.index = self.dense_index

        return unique_contexts, dense_index, bm25_index

    def reciprocal_rank_fusion(self, dense_ranks: list[int], sparse_ranks: list[int]) -> list[int]:
        """Fuses dense and sparse rank lists using Weighted RRF.
        Returns the FULL fused ranking (truncation happens in retrieve())."""
        scores: dict[int, float] = {}
        for rank, doc_id in enumerate(dense_ranks):
            scores[doc_id] = scores.get(doc_id, 0.0) + self.dense_weight / (self.rrf_k + rank + 1)
        for rank, doc_id in enumerate(sparse_ranks):
            scores[doc_id] = scores.get(doc_id, 0.0) + self.sparse_weight / (self.rrf_k + rank + 1)
        sorted_docs = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        return [doc_id for doc_id, _ in sorted_docs]

    def get_fused_ranking(self, question: str) -> list[int]:
        """Stage-1: dense + sparse retrieval, fused with weighted RRF.
        Single source of truth for HybridRAG and AdvancedHybridRAG —
        guarantees the reranker is the ONLY delta between the two systems."""
        # 1. Dense Retrieval
        q_emb = self.embedder.encode([question], normalize_embeddings=True)
        _, dense_indices = self.dense_index.search(np.array(q_emb, dtype=np.float32), self.top_k_dense)
        dense_ranked_ids = dense_indices[0].tolist()

        # 2. Sparse Retrieval
        tokenized_query = question.lower().split(" ")
        sparse_scores = self.bm25_index.get_scores(tokenized_query)
        sparse_ranked_ids = np.argsort(sparse_scores)[::-1][: self.top_k_sparse].tolist()

        # 3. Hybrid Fusion via RRF
        return self.reciprocal_rank_fusion(dense_ranked_ids, sparse_ranked_ids)

    def retrieve(self, question: str, k: int | None = None) -> list[str]:
        k = k if k is not None else self.top_k
        retrieved_ids = self.get_fused_ranking(question)[:k]
        retrieved_chunks = [self.unique_contexts[idx] for idx in retrieved_ids]
        return retrieved_chunks