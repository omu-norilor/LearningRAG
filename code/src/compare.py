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
from constants import HF_TOKEN
SETTING = 3

# --- Execution Script ---

# Load 150 answerable questions for testing
ds = load_dataset("rajpurkar/squad_v2", split="validation", token=HF_TOKEN)
answerable_ds = ds.filter(lambda x: len(x["answers"]["text"]) > 0).select(range(150))



# ------ Test cases ------
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

    evaluate(naive_llama32_3b, answerable_ds, "Naive RAG (Dense) - LLaMA 3.2:3B")
    evaluate(naive_llama31_8b, answerable_ds, "Naive RAG (Dense) - LLaMA 3.1:8B")
    evaluate(naive_qwen3_8b, answerable_ds, "Naive RAG (Dense) - Qwen3:8B")
    evaluate(naive_qwen25_7b, answerable_ds, "Naive RAG (Dense) - Qwen2.5:7B")


if SETTING == 2:
    naive = NaiveRAG(
    dataset="rajpurkar/squad_v2",
    split="validation",
    top_k=5,
    model_name="qwen3:8b",
    )

    hybrid = HybridRAG(
    dataset="rajpurkar/squad_v2", 
    split="validation", 
    top_k=5,
    model_name="qwen3:8b",
    top_k_dense = 50,     # <-- INCREASED from 20 to 50
    top_k_sparse = 50,    # <-- INCREASED from 20 to 50
    )

    evaluate(naive, answerable_ds, "Naive RAG (Dense)")
    evaluate(hybrid, answerable_ds, "Hybrid RAG (BM25 + Dense)")


if SETTING == 3:
    hybrid = HybridRAG(
    dataset="rajpurkar/squad_v2", 
    split="validation", 
    top_k=5,
    model_name="qwen3:8b",
    top_k_dense = 50,     # <-- INCREASED from 20 to 50
    top_k_sparse = 50,    # <-- INCREASED from 20 to 50
    )

    advanced_hybrid = AdvancedHybridRAG(
    dataset="rajpurkar/squad_v2",
    split="validation",
    top_k=5,
    model_name="qwen3:8b",
    top_k_dense = 50,     # <-- INCREASED from 20 to 50
    top_k_sparse = 50,    # <-- INCREASED from 20 to 50
    )

    evaluate(hybrid, answerable_ds, "Hybrid RAG (BM25 + Dense)")
    evaluate(advanced_hybrid, answerable_ds, "Advanced Hybrid RAG (Cross-Encoder)")

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
    evaluate(naive, answerable_ds, "Naive RAG (Dense)")
    evaluate(HyDE, answerable_ds, "HyDE RAG (Dense + Hypotohetical Document Embeddings)")

# Till now, NaiveRAG wins, because the documents do not contain keywords, so BM25 does not help.
# Further, the cross-encoder in AdvancedHybridRAG does not help, as the sparse search is already misleading.
# Hyde did better in terms of recall, but worse when at extraction, yielding lower EM/F1.