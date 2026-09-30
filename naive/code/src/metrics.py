import os
import sys
import string
import re
import numpy as np
from datasets import load_dataset
from evaluate import load



# --- 1. Text Normalization & Extraction Metrics ---

def normalize_text(s: str) -> str:
    """Lowercases, removes punctuation, articles, and extra whitespace."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)
    def white_space_fix(text):
        return ' '.join(text.split())
    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)
    
    return white_space_fix(remove_articles(remove_punc(s.lower())))


def compute_token_f1(prediction: str, gold_answers: list[str]) -> float:
    """Computes token-level F1 score between prediction and gold answers."""
    pred_tokens = normalize_text(prediction).split()
    if not pred_tokens:
        return 0.0

    f1s = []
    for gold in gold_answers:
        gold_tokens = normalize_text(gold).split()
        if not gold_tokens:
            continue
            
        common = set(pred_tokens) & set(gold_tokens)
        if not common:
            f1s.append(0.0)
            continue
            
        precision = len(common) / len(pred_tokens)
        recall = len(common) / len(gold_tokens)
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        f1s.append(f1)
        
    return max(f1s) if f1s else 0.0


# --- 2. Retrieval Metrics ---

def compute_retrieval_metrics(retrieved_chunks: list[str], gold_context: str, k_values: list[int] = [1, 3, 5]) -> dict:
    """Computes Recall@K and Mean Reciprocal Rank (MRR) for retrieved chunks."""
    metrics = {}
    
    # Check which ranks contain the gold context
    gold_rank = -1
    for idx, chunk in enumerate(retrieved_chunks):
        if gold_context.strip() in chunk.strip() or chunk.strip() in gold_context.strip():
            gold_rank = idx + 1
            break
            
    # Recall@K
    for k in k_values:
        metrics[f"recall@{k}"] = 1.0 if (gold_rank != -1 and gold_rank <= k) else 0.0
        
    # MRR
    metrics["mrr"] = (1.0 / gold_rank) if gold_rank != -1 else 0.0
    return metrics


# --- 3. Diagnostic & Classification Metric ---

def diagnose_failure(retrieved_correctly: bool, exact_match: bool) -> str:
    """Categorizes the sample failure type."""
    if not retrieved_correctly:
        return "Retrieval Miss"
    elif not exact_match:
        return "Generation/Extraction Miss"
    return "Success"


# --- 4. Master Evaluation Function ---

def evaluate(rag_instance, dataset, name="RAG"):
    print(f"\n--- Running Comprehensive Evaluation for {name} ---")
    
    squad_metric = load("squad_v2")
    
    predictions = []
    references = []
    
    total_samples = len(dataset)
    retrieval_recalls = {1: 0.0, 3: 0.0, 5: 0.0}
    mrr_total = 0.0
    f1_scores = []
    failure_counts = {"Success": 0, "Retrieval Miss": 0, "Generation/Extraction Miss": 0}
    
    for sample in dataset:
        # Run RAG
        prompt, answer, retrieved_chunks = rag_instance.run(sample["question"])
        
        # Collect for HuggingFace SQuAD evaluation
        predictions.append({
            "id": sample["id"],
            "prediction_text": answer,
            "no_answer_probability": 0.0
        })
        references.append({
            "id": sample["id"],
            "answers": sample["answers"]
        })
        
        # 1. Extraction Metrics (F1)
        gold_answers = sample["answers"]["text"]
        sample_f1 = compute_token_f1(answer, gold_answers)
        f1_scores.append(sample_f1)
        
        # 2. Retrieval Metrics
        gold_context = sample["context"]
        ret_metrics = compute_retrieval_metrics(retrieved_chunks, gold_context, k_values=[1, 3, 5])
        retrieval_recalls[1] += ret_metrics["recall@1"]
        retrieval_recalls[3] += ret_metrics["recall@3"]
        retrieval_recalls[5] += ret_metrics["recall@5"]
        mrr_total += ret_metrics["mrr"]
        
        # 3. Diagnostic Categorization
        # Check standard normalized match for diagnostic purposes
        norm_pred = normalize_text(answer)
        is_em = any(norm_pred == normalize_text(g) for g in gold_answers)
        retrieved_correctly = ret_metrics["recall@3"] > 0.0  # evaluated based on top_k=3
        
        failure_type = diagnose_failure(retrieved_correctly, is_em)
        failure_counts[failure_type] += 1

    # Compute official metrics via HuggingFace
    squad_results = squad_metric.compute(predictions=predictions, references=references)
    
    # Average out custom metrics
    avg_f1_custom = np.mean(f1_scores) * 100
    avg_mrr = (mrr_total / total_samples)
    avg_recall_1 = (retrieval_recalls[1] / total_samples) * 100
    avg_recall_3 = (retrieval_recalls[3] / total_samples) * 100
    avg_recall_5 = (retrieval_recalls[5] / total_samples) * 100

    print(f"\n[Generation / Extraction Metrics]")
    print(f"  Exact Match: {squad_results['exact']:.2f}% | HF F1: {squad_results['f1']:.2f}% | Custom Token F1: {avg_f1_custom:.2f}%")
    
    print(f"\n[Retrieval Metrics]")
    print(f"  Recall@1: {avg_recall_1:.2f}% | Recall@3: {avg_recall_3:.2f}% | Recall@5: {avg_recall_5:.2f}% | MRR: {avg_mrr:.4f}")
    
    print(f"\n[Diagnostic Breakdown]")
    for category, count in failure_counts.items():
        pct = (count / total_samples) * 100
        print(f"  {category}: {count}/{total_samples} ({pct:.1f}%)")
    print("-" * 50)
    
    return {
        "squad_results": squad_results,
        "custom_f1": avg_f1_custom,
        "recall_at_3": avg_recall_3,
        "mrr": avg_mrr,
        "diagnostics": failure_counts
    }
