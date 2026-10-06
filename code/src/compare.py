import os
import sys
import string
import re
import numpy as np
from datasets import load_dataset
from evaluate import load
from rag_classes.naive import NaiveRAG
from rag_classes.hybrid import HybridRAG
from rag_classes.hybrid_advanced import AdvancedHybridRAG
from rag_classes.HyDE import HyDERAG
from metrics import evaluate
from utils import build_balanced_eval_set
from constants import HF_TOKEN


# --- Execution Script ---

# Load 150 answerable questions for testing
balanced_ds = build_balanced_eval_set(n_per_class=100, seed=42)   # same set for both models

# ------ Test cases ------
SETTING = 3
if SETTING == 1:
    # Compare LLM models within the NaiveRAG framework
    naive_llama32_3b = NaiveRAG(
        dataset="rajpurkar/squad_v2",
        split="validation",
        top_k=5,
        model_name="llama3.2:3b",
    )

    naive_llama31_8b = NaiveRAG(
        dataset="rajpurkar/squad_v2",
        split="validation",
        top_k=5,
        model_name="llama3.1:8b",
    )

    naive_qwen3_8b = NaiveRAG(
        dataset="rajpurkar/squad_v2",
        split="validation",
        top_k=5,
        model_name="qwen3:8b",
    )

    naive_qwen25_7b = NaiveRAG(
        dataset="rajpurkar/squad_v2",
        split="validation",
        top_k=5,
        model_name="qwen2.5:7b",
    )

    evaluate(naive_llama32_3b, balanced_ds, "Naive RAG (Dense) - LLaMA 3.2:3B")
    evaluate(naive_llama31_8b, balanced_ds, "Naive RAG (Dense) - LLaMA 3.1:8B")
    evaluate(naive_qwen3_8b, balanced_ds, "Naive RAG (Dense) - Qwen3:8B")
    evaluate(naive_qwen25_7b, balanced_ds, "Naive RAG (Dense) - Qwen2.5:7B")


if SETTING == 2:
    naive = NaiveRAG(
        dataset="rajpurkar/squad_v2",
        split="validation",
        extra_split="train[:25000]",
        top_k=5,
        model_name="qwen3:8b",
    )

    hybrid = HybridRAG(
        dataset="rajpurkar/squad_v2", 
        split="validation",
        extra_split="train[:25000]",
        top_k=5,
        model_name="qwen3:8b",
        top_k_dense = 50,
        top_k_sparse = 50,
    )

    evaluate(naive, balanced_ds, "Naive RAG (Dense)")
    evaluate(hybrid, balanced_ds, "Hybrid RAG (BM25 + Dense)")


if SETTING == 3:
    hybrid = HybridRAG(
        dataset="rajpurkar/squad_v2", 
        split="validation",
        extra_split="train[:25000]",
        top_k=5,
        model_name="qwen3:8b",
        top_k_dense = 50,
        top_k_sparse = 50,
    )

    advanced_hybrid = AdvancedHybridRAG(
        dataset="rajpurkar/squad_v2",
        split="validation",
        extra_split="train[:25000]",
        top_k=5,
        model_name="qwen3:8b",
        top_k_dense = 50,
        top_k_sparse = 50,
        rerank_depth = 30,
        reranker_model="cross-encoder/ms-marco-MiniLM-L-6-v2",
    )

    evaluate(hybrid, balanced_ds, "Hybrid RAG (BM25 + Dense)")
    evaluate(advanced_hybrid, balanced_ds, "Advanced Hybrid RAG (Cross-Encoder)")

if SETTING == 4:
    naive = HybridRAG(
        dataset="rajpurkar/squad_v2",
        split="validation",
        top_k=5,
        model_name="qwen3:8b",
    )

    HyDE = HyDERAG(
        dataset="rajpurkar/squad_v2",
        split="validation",
        top_k=5,
        model_name="qwen3:8b",
    )
    evaluate(naive, balanced_ds, "Naive RAG (Dense)")
    evaluate(HyDE, balanced_ds, "HyDE RAG (Dense + Hypotohetical Document Embeddings)")

# Till now, NaiveRAG wins, because the documents do not contain keywords, so BM25 does not help.
# Further, the cross-encoder in AdvancedHybridRAG does not help, as the sparse search is already misleading.
# Hyde did better in terms of recall, but worse when at extraction, yielding lower EM/F1.