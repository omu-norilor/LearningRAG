import json
import os
import faiss
import numpy as np
import ollama
from datasets import load_dataset
from evaluate import load
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------
# 1. SETUP & CONFIG
# ---------------------------------------------------------
EVAL_SAMPLES = 150
TOP_K = 3
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://ollama-service:11434")
OUTPUT_FAILURE_PATH = "/app/data/failure_analysis.json"

print(f"Loading SQuAD 2.0 validation split (First {EVAL_SAMPLES} samples)...")
val_dataset = load_dataset("rajpurkar/squad_v2", split=f"validation[:{EVAL_SAMPLES}]")
squad_metric = load("squad_v2")

# De-duplicate unique contexts for corpus building
unique_contexts = list(dict.fromkeys(val_dataset["context"]))
context_to_id = {ctx: i for i, ctx in enumerate(unique_contexts)}
print(f"Indexed {len(unique_contexts)} unique context paragraphs.")

# ---------------------------------------------------------
# 2. VECTOR INDEX CONSTRUCTION
# ---------------------------------------------------------
print("Generating corpus embeddings (BAAI/bge-small-en-v1.5)...")
embedder = SentenceTransformer("BAAI/bge-small-en-v1.5")
corpus_embeddings = embedder.encode(unique_contexts, normalize_embeddings=True)

dimension = corpus_embeddings.shape[1]
index = faiss.IndexFlatIP(dimension)  # Cosine similarity via Inner Product on unit vectors
index.add(np.array(corpus_embeddings, dtype=np.float32))

client = ollama.Client(host=OLLAMA_HOST)


def raw_generate(prompt: str) -> str:
    """Executes a raw completion without chat wrappers."""
    response = client.generate(
        model="llama3.2:3b",
        prompt=prompt,
        raw=True,
        options={"temperature": 0.0, "num_predict": 64},
    )
    return response["response"].strip()


# ---------------------------------------------------------
# 3. EVALUATION LOOP
# ---------------------------------------------------------
print("\nStarting evaluation run...")

recalls = []
mrr_list = []
predictions = []
references = []
failures = []

for idx, sample in enumerate(val_dataset):
    q_id = sample["id"]
    question = sample["question"]
    gold_context = sample["context"]
    gold_answers = sample["answers"]
    gold_context_id = context_to_id[gold_context]

    # --- RETRIEVAL ---
    q_emb = embedder.encode([question], normalize_embeddings=True)
    _, top_k_indices = index.search(np.array(q_emb, dtype=np.float32), TOP_K)
    retrieved_indices = top_k_indices[0].tolist()
    retrieved_chunks = [unique_contexts[i] for i in retrieved_indices]

    # Retrieval Metrics (Recall@k & MRR)
    hit = gold_context_id in retrieved_indices
    recalls.append(1.0 if hit else 0.0)

    if hit:
        rank = retrieved_indices.index(gold_context_id) + 1
        mrr_list.append(1.0 / rank)
    else:
        mrr_list.append(0.0)

    # --- GENERATION ---
    context_str = "\n\n".join(retrieved_chunks)
    # prompt = (
    #     f"Context:\n{context_str}\n\n"
    #     f"Answer the question using ONLY the context provided. "
    #     f"If the answer cannot be found in the context, write 'Unanswerable'.\n\n"
    #     f"Question: {question}\nAnswer:"
    # )
    prompt = (
        f"Context:\n{context_str}\n\n"
        f"Instructions:\n"
        f"- Answer the question using ONLY an exact short phrase or word directly from the context.\n"
        f"- If the question cannot be answered using the context, respond with EXACTLY 'Unanswerable'.\n"
        f"- Do not write full sentences or explanations.\n\n"
        f"Question: {question}\n"
        f"Answer:"
    )

    raw_output = raw_generate(prompt)
    generated_answer = clean_completion(raw_output)

    # Prepare for HF Evaluate squad_v2 format
    predictions.append({"id": q_id, "prediction_text": generated_answer, "no_answer_probability": 0.0})
    references.append(
        {
            "id": q_id,
            "answers": {
                "text": gold_answers["text"],
                "answer_start": gold_answers["answer_start"],
            },
        }
    )

    # --- FAILURE LOGGING ---
    # Quick exact match check against gold string targets
    is_exact_match = any(generated_answer.lower() == ans.lower() for ans in gold_answers["text"])
    if len(gold_answers["text"]) == 0 and generated_answer.lower() == "unanswerable":
        is_exact_match = True

    if not is_exact_match:
        failure_type = "Retrieval Miss" if not hit else "Generation Miss"
        failures.append(
            {
                "sample_id": q_id,
                "failure_type": failure_type,
                "question": question,
                "gold_answers": gold_answers["text"],
                "generated_answer": generated_answer,
                "gold_context_retrieved": hit,
                "retrieved_chunks": retrieved_chunks,
            }
        )

    if (idx + 1) % 25 == 0 or (idx + 1) == EVAL_SAMPLES:
        print(f"Processed {idx + 1}/{EVAL_SAMPLES} samples...")

# ---------------------------------------------------------
# 4. METRIC COMPUTATION & FAILURE ANALYSIS DUMP
# ---------------------------------------------------------
results = squad_metric.compute(predictions=predictions, references=references)

mean_recall = np.mean(recalls) * 100
mean_mrr = np.mean(mrr_list)

print("\n" + "=" * 50)
print("             EVALUATION RESULTS (LEVEL 1)          ")
print("=" * 50)
print(f"Retrieval Recall@{TOP_K} : {mean_recall:.2f}%")
print(f"Retrieval MRR       : {mean_mrr:.4f}")
print("-" * 50)
print(f"Exact Match (EM)    : {results['exact']:.2f}%")
print(f"F1 Score            : {results['f1']:.2f}%")
print(f"HasAns EM           : {results.get('HasAns_exact', 0.0):.2f}%")
print(f"NoAns EM            : {results.get('NoAns_exact', 0.0):.2f}%")
print("=" * 50)

# Save failure log for inspection
os.makedirs(os.path.dirname(OUTPUT_FAILURE_PATH), exist_ok=True)
with open(OUTPUT_FAILURE_PATH, "w") as f:
    json.dump(failures, f, indent=2)

retrieved_misses = sum(1 for f in failures if f["failure_type"] == "Retrieval Miss")
generation_misses = sum(1 for f in failures if f["failure_type"] == "Generation Miss")

print(f"\nSaved {len(failures)} failure cases to `{OUTPUT_FAILURE_PATH}`")
print(f"  └── Retrieval Misses  : {retrieved_misses}")
print(f"  └── Generation Misses : {generation_misses}\n")