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
from metrics import evaluate



# --- Execution Script ---
# Load 50 answerable questions for testing
ds = load_dataset("rajpurkar/squad_v2", split="validation")
answerable_ds = ds.filter(lambda x: len(x["answers"]["text"]) > 0).select(range(50))

naive = NaiveRAG(dataset="rajpurkar/squad_v2", split="validation", top_k=3)
hybrid = HybridRAG(dataset="rajpurkar/squad_v2", split="validation", top_k=3)
advanced_hybrid = AdvancedHybridRAG(dataset="rajpurkar/squad_v2", split="validation", top_k=3)

evaluate(naive, answerable_ds, "Naive RAG (Dense)")
evaluate(hybrid, answerable_ds, "Hybrid RAG (BM25 + Dense)")
evaluate(advanced_hybrid, answerable_ds, "Advanced Hybrid RAG (Cross-Encoder)")