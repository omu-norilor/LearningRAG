import os
import sys
import faiss
import numpy as np
import ollama
import random
from datasets import load_dataset
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi
from .hybrid import HybridRAG

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# append sys path to locate the 'src' package
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from constants import HF_TOKEN


class AdvancedHybridRAG(HybridRAG):
    def __init__(
        self,
        top_k: int = 3,
        top_k_dense: int = 30,
        top_k_sparse: int = 30,
        rrf_k: int = 60,
        dense_weight: float = 0.7,
        sparse_weight: float = 0.3,
        rerank_depth: int = 30,
        dataset: str = "rajpurkar/squad_v2",
        split: str = "validation[:200]",
        model_name: str = "llama3.2:3b",
        embedding_model: str = "BAAI/bge-small-en-v1.5",
        extra_split: str | None = None,
        reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    ):
        # set reranker-specific params before parent initializers run
        self.rerank_depth = rerank_depth          # Fix 3: stage-1 pool = the ceiling
        self.reranker_model = reranker_model

        # parent HybridRAG builds dense + sparse indexes and the corpus (incl. extra_split)
        super().__init__(
            top_k=top_k,
            top_k_dense=top_k_dense,
            top_k_sparse=top_k_sparse,
            rrf_k=rrf_k,
            dense_weight=dense_weight,
            sparse_weight=sparse_weight,
            dataset=dataset,
            split=split,
            model_name=model_name,
            embedding_model=embedding_model,
            extra_split=extra_split,              # Fix 1: forwarded
        )

        print(f"Loading Cross-Encoder ({self.reranker_model})...")
        self.reranker = CrossEncoder(self.reranker_model)

    def retrieve(self, question: str, k: int | None = None) -> list[str]:
        k = k if k is not None else self.top_k

        # Stage 1 — the SAME fusion as the baseline, truncated to the reranker's pool
        candidate_ids = self.get_fused_ranking(question)[: self.rerank_depth]
        self.last_candidate_ids = candidate_ids   # exposed for stage-1 ceiling metrics

        # Stage 2 — cross-encoder reads (question, paragraph) pairs and re-ranks
        pairs = [[question, self.unique_contexts[doc_id]] for doc_id in candidate_ids]
        scores = self.reranker.predict(pairs, batch_size=32)

        order = np.argsort(scores)[::-1][:k]
        return [self.unique_contexts[candidate_ids[i]] for i in order]

