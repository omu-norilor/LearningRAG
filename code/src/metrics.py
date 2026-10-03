# metrics.py
import os
import sys
import string
import re
import numpy as np
from datasets import load_dataset
from evaluate import load

ANSWERABLE, UNANSWERABLE = "answerable", "unanswerable"


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


def is_no_answer(answer: str) -> bool:
    """True if the model output signals abstention ('Unanswerable' sentinel or refusal phrasing)."""
    text = normalize_text(answer or "").replace("_", " ").strip()
    if not text:
        return True  # empty output counts as abstention
    return any(re.search(p, text) for p in _ABSTENTION_PATTERNS)


_ABSTENTION_PATTERNS = [
    r"\bno ?answer\b", r"\bnot ?answerable\b", r"\bunanswerable\b",
    r"\b(cannot|can ?t|could ?not) (answer|be answered|find|determine|locate)\b",
    r"\b(don ?t|does ?not|doesnt) (know|mention|specify|provide|state|say|contain)\b",
    r"\bnot (mentioned|specified|provided|stated|found|contained|available|given)\b",
    r"\bnot (enough|sufficient) (information|context|detail)\b",
    r"\bno (information|relevant information|details)\b",
    r"\bnot in (the |this )?(context|passage|text|document)\b",
    r"\bunable to (find|determine|answer)\b", r"\bunknown\b",
]


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

def diagnose_failure(is_answerable: bool, retrieved_correctly: bool, f1: float, abstained: bool,
                     f1_threshold: float = 0.5) -> str:
    """Categorizes the sample failure type. Success = retrieval hit + token F1 >= threshold."""
    if is_answerable:
        if not retrieved_correctly:
            return "Retrieval Miss"
        if f1 >= f1_threshold:
            return "Success"
        return "False Abstention" if abstained else "Generation/Extraction Miss"
    return "Correct Abstention" if abstained else "False Answer"


# --- 4. Master Evaluation Function ---

def evaluate(rag_instance, dataset, name="RAG"):
    print(f"\n--- Running Comprehensive Evaluation for {name} ---")
    
    squad_metric = load("squad_v2")
    
    predictions = []
    references = []
    
    total_samples = len(dataset)
    retrieval_recalls = {1: 0.0, 3: 0.0, 5: 0.0}
    mrr_total = 0.0
    em_scores = {ANSWERABLE: [], UNANSWERABLE: []}
    f1_scores = {ANSWERABLE: [], UNANSWERABLE: []}
    failure_counts = {"Success": 0, "Retrieval Miss": 0, "Generation/Extraction Miss": 0,
                      "False Abstention": 0, "Correct Abstention": 0, "False Answer": 0}
    
    for sample in dataset:
        # Run RAG
        prompt, answer, retrieved_chunks = rag_instance.run(sample["question"])

        gold_answers = sample["answers"]["text"]
        is_answerable = len(gold_answers) > 0
        cls = ANSWERABLE if is_answerable else UNANSWERABLE
        abstained = is_no_answer(answer)
        
        # Collect for HuggingFace SQuAD evaluation
        predictions.append({
            "id": sample["id"],
            "prediction_text": "" if abstained else answer,
            "no_answer_probability": 1.0 if abstained else 0.0
        })
        references.append({
            "id": sample["id"],
            "answers": sample["answers"]
        })
        
        # 1. Extraction Metrics (EM / F1, per class)
        if is_answerable:
            sample_f1 = compute_token_f1(answer, gold_answers)
            norm_pred = normalize_text(answer)
            is_em = any(norm_pred == normalize_text(g) for g in gold_answers)
        else:
            sample_f1 = 1.0 if abstained else 0.0
            is_em = abstained
        em_scores[cls].append(float(is_em))
        f1_scores[cls].append(sample_f1)
        
        # 2. Retrieval Metrics
        gold_context = sample["context"]
        ret_metrics = compute_retrieval_metrics(retrieved_chunks, gold_context, k_values=[1, 3, 5])
        retrieval_recalls[1] += ret_metrics["recall@1"]
        retrieval_recalls[3] += ret_metrics["recall@3"]
        retrieval_recalls[5] += ret_metrics["recall@5"]
        mrr_total += ret_metrics["mrr"]
        
        # 3. Diagnostic Categorization
        retrieved_correctly = ret_metrics["recall@3"] > 0.0  # evaluated based on top_k=3
        failure_type = diagnose_failure(is_answerable, retrieved_correctly, sample_f1, abstained)
        failure_counts[failure_type] += 1

    # Compute official metrics via HuggingFace
    try:
        squad_results = squad_metric.compute(predictions=predictions, references=references,
                                             no_answer_threshold=0.5)
    except TypeError:  # older `evaluate` versions without the kwarg
        squad_results = squad_metric.compute(predictions=predictions, references=references)
    
    # Average out custom metrics
    avg_f1_custom = np.mean(f1_scores[ANSWERABLE] + f1_scores[UNANSWERABLE]) * 100
    avg_mrr = (mrr_total / total_samples)
    avg_recall_1 = (retrieval_recalls[1] / total_samples) * 100
    avg_recall_3 = (retrieval_recalls[3] / total_samples) * 100
    avg_recall_5 = (retrieval_recalls[5] / total_samples) * 100

    def mean(xs):
        return float(np.mean(xs)) if xs else 0.0

    print(f"\n[Generation / Extraction Metrics]")
    for c in (ANSWERABLE, UNANSWERABLE):
        label = "Answerable" if c == ANSWERABLE else "Unanswerable"
        print(f"  {label:<14} (n={len(f1_scores[c])}): EM {100 * mean(em_scores[c]):6.2f}% | Token F1 {100 * mean(f1_scores[c]):6.2f}%")
    print(f"  {'Overall (official)':<14}: EM {squad_results['exact']:.2f}% | HF F1 {squad_results['f1']:.2f}% | Custom Token F1: {avg_f1_custom:.2f}%")
    print(f"  {'':<14}  HasAns EM/F1: {squad_results['HasAns_exact']:.2f}/{squad_results['HasAns_f1']:.2f} | NoAns EM/F1: {squad_results['NoAns_exact']:.2f}/{squad_results['NoAns_f1']:.2f}")
    
    print(f"\n[Retrieval Metrics]")
    print(f"  Recall@1: {avg_recall_1:.2f}% | Recall@3: {avg_recall_3:.2f}% | Recall@5: {avg_recall_5:.2f}% | MRR: {avg_mrr:.4f}")
    
        # per-class denominators: each category only occurs within one class
    class_n = {ANSWERABLE: len(f1_scores[ANSWERABLE]), UNANSWERABLE: len(f1_scores[UNANSWERABLE])}
    diag_class = {"Success": ANSWERABLE, "Retrieval Miss": ANSWERABLE,
                  "Generation/Extraction Miss": ANSWERABLE, "False Abstention": ANSWERABLE,
                  "Correct Abstention": UNANSWERABLE, "False Answer": UNANSWERABLE}

    print(f"\n[Diagnostic Breakdown]")
    diag_groups = (
        ("Answerable", ("Success", "Retrieval Miss", "Generation/Extraction Miss", "False Abstention"),
         len(f1_scores[ANSWERABLE])),
        ("Unanswerable", ("Correct Abstention", "False Answer"),
         len(f1_scores[UNANSWERABLE])),
    )
    for label, categories, n in diag_groups:
        print(f"  {label} (n={n})")
        for category in categories:
            count = failure_counts.get(category, 0)
            print(f"    {category}: {count}/{n} ({100 * count / max(n, 1):.1f}%)")
    print("-" * 50)
    
    return {
        "squad_results": squad_results,
        "custom_f1": avg_f1_custom,
        "recall_at_3": avg_recall_3,
        "mrr": avg_mrr,
        "diagnostics": failure_counts,
        "per_class": {
            "answerable": {"n": len(em_scores[ANSWERABLE]), "em": mean(em_scores[ANSWERABLE]), "f1": mean(f1_scores[ANSWERABLE])},
            "unanswerable": {"n": len(em_scores[UNANSWERABLE]), "em": mean(em_scores[UNANSWERABLE]), "f1": mean(f1_scores[UNANSWERABLE])},
        },
    }