# code/src/compare_rag.py
import numpy as np
from datasets import load_dataset
from evaluate import load
from rag import NaiveRAG, HybridRAG

# Load 150 answerable questions
ds = load_dataset("rajpurkar/squad_v2", split="validation")
answerable_ds = ds.filter(lambda x: len(x["answers"]["text"]) > 0).select(range(150))

squad_metric = load("squad_v2")

def evaluate_rag(rag_instance, dataset, name="RAG"):
    predictions = []
    references = []
    
    print(f"\n--- Running Evaluation for {name} ---")
    for sample in dataset:
        prompt, answer, chunks = rag_instance.run(sample["question"])
        
        predictions.append({
            "id": sample["id"],
            "prediction_text": answer,
            "no_answer_probability": 0.0
        })
        references.append({
            "id": sample["id"],
            "answers": sample["answers"]
        })
        
    results = squad_metric.compute(predictions=predictions, references=references)
    print(f"[{name}] Exact Match: {results['exact']:.2f}% | F1: {results['f1']:.2f}%")
    return results

naive = NaiveRAG()
hybrid = HybridRAG()

evaluate_rag(naive, answerable_ds, "Naive RAG (Dense)")
evaluate_rag(hybrid, answerable_ds, "Hybrid RAG (BM25 + Dense)")