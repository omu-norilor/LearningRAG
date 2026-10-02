import os
import sys
import json
import faiss
import numpy as np
import ollama
import random
from datasets import load_dataset
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi

# append sys path to locate the 'src' package
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from constants import HF_TOKEN


class BaseRAG:
    """Shared base class for RAG variants.

    Responsibilities:
    - initialize common clients/embedders
    - call subclass `build_index()` during init
    - provide `raw_generate`, `clean_completion`, `assemble_prompt`, `retrieve` (default dense), and `run`
    """

    def __init__(
        self,
        top_k: int = 3,
        dataset: str = "rajpurkar/squad_v2",
        split: str = "validation[:200]",
        model_name: str = "llama3.2:3b",
        embedding_model: str = "BAAI/bge-small-en-v1.5",
    ):
        # Force offline mode to use local cache and avoid name resolution timeouts
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"

        self.top_k = top_k
        self.client = ollama.Client(host=os.getenv("OLLAMA_HOST", "http://ollama-service:11434"))
        
        # Load embedder with local_files_only=True to strictly hit cache
        self.embedder = SentenceTransformer(embedding_model, local_files_only=True, token=HF_TOKEN)

        self.dataset = dataset
        self.split = split
        self.model_name = model_name

        # Subclass must implement build_index()
        self.build_index()

    def build_index(self):
        raise NotImplementedError("Subclasses must implement build_index()")

    def raw_generate(self, system: str = "", prompt: str = "") -> str:
        response = self.client.generate(
            model=self.model_name,
            prompt=prompt,
            system=system,
            think=False,
            format={                          # <-- HERE, top-level
                "type": "object",
                "properties": {"answer": {"type": "string"}},
                "required": ["answer"],
            },
            options={"temperature": 0.0, "num_predict": 256, "num_ctx": 8192},
        )
        try:
            return json.loads(response["response"])["answer"]
        except (json.JSONDecodeError, KeyError):
            return response["response"]       # fallback: let post-processor handle raw text

    def clean_completion(self, text: str) -> str:
        lines = [line.strip() for line in text.strip().split("\n") if line.strip()]
        if not lines:
            return ""
        text = lines[0]

        lowercased = text.lower()
        for prefix in ["answer:", "the answer is", "based on the context,"]:
            if lowercased.startswith(prefix):
                text = text[len(prefix) :].strip()
                lowercased = text.lower()

        if lowercased.startswith("unanswerable") or "not provided" in lowercased or "cannot be found" in lowercased:
            return "Unanswerable"

        return text.rstrip(".")

    def assemble_prompt(self, retrieved_chunks: list[str], question: str) -> (str, str):
        context_str = "\n\n".join(retrieved_chunks)


        system = (
            "You are an extractive QA system. Respond with ONLY the answer span, "
            "copied verbatim from the context. No explanations, no quotes, no notes, "
            "no parentheses. If the context does not contain the answer, respond "
            "with only: Unanswerable"
        )

        prompt = (
            f"Context:\n{context_str}\n\n"
            f"Question: {question}\n\n"
            "Correct response examples:\n"
            "Question: Where were the matches played? -> Berlin\n"
            "Question: What year did the conflict end? -> 1945\n"
            "Question: How many spectators attended? -> Unanswerable\n\n"
            "Answer with only the exact span (or 'Unanswerable'):"
        )

        # prompt = (
        #     f"Context:\n{context_str}\n\n"
        #     f"Instructions:\n"
        #     f"1. Answer the question using ONLY an exact word or short phrase directly from the context.\n"
        #     f"2. If the context does not explicitly contain the answer, respond with EXACTLY 'Unanswerable'.\n"
        #     f"3. Do not guess or use outside knowledge.\n\n"
        #     f"Question: {question}\n"
        #     f"Answer:"
        # )
        return system, prompt

    def retrieve(self, question: str, k: int | None = None) -> list[str]:
        """Default dense retrieval using `self.dense_index`.

        Returns a list of retrieved context strings (length `k` or `self.top_k`).
        """
        if k is None:
            k = self.top_k

        q_emb = self.embedder.encode([question], normalize_embeddings=True)
        _, indices = self.dense_index.search(np.array(q_emb, dtype=np.float32), k)
        retrieved_chunks = [self.unique_contexts[idx] for idx in indices[0]]
        return retrieved_chunks

    def run(self, question: str, top_k: int | None = None):
        k = top_k if top_k is not None else self.top_k
        retrieved_chunks = self.retrieve(question, k=k)
        system, prompt = self.assemble_prompt(retrieved_chunks, question)
        raw_output = self.raw_generate(system ,prompt)
        answer = self.clean_completion(raw_output)
        return prompt, answer, retrieved_chunks