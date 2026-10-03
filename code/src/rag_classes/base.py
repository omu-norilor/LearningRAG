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

    def raw_generate(self, system: str = "", prompt: str = "") -> dict:
        response = self.client.generate(
            model=self.model_name,
            prompt=prompt,
            system=system,
            think=False,
            format={  # abstention becomes an explicit boolean, not string matching
                "type": "object",
                "properties": {
                    "answerable": {"type": "boolean"},   # listed first: decide BEFORE writing a span
                    "answer": {"type": "string"},
                },
                "required": ["answerable", "answer"],
            },
            options={"temperature": 0.0, "num_predict": 256, "num_ctx": 8192},
        )
        text = response["response"]
        try:
            parsed = json.loads(text)
            return parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            # salvage fields instead of leaking raw JSON into clean_completion
            flag = re.search(r'"answerable"\s*:\s*(true|false)', text, re.IGNORECASE)
            ans = re.search(r'"answer"\s*:\s*"((?:[^"\\]|\\.)*)"', text)
            if ans:
                return {"answerable": flag.group(1).lower() == "true" if flag else True,
                        "answer": ans.group(1)}
            return {"answerable": True, "answer": text}


    def clean_completion(self, raw) -> str:
        # raw_generate now returns a dict; str accepted for legacy/subclass overrides
        if isinstance(raw, dict):
            answerable = raw.get("answerable", True)
            text = (raw.get("answer") or "").strip()
        else:
            answerable, text = True, (raw or "").strip()

        lines = [line.strip() for line in text.split("\n") if line.strip()]
        text = lines[0] if lines else ""

        lowercased = text.lower()
        for prefix in ["answer:", "the answer is", "based on the context,", "according to the context,"]:
            if lowercased.startswith(prefix):
                text = text[len(prefix):].strip()
                lowercased = text.lower()

        # the flag is authoritative; text triggers remain as fallback for the legacy path
        if answerable is False or lowercased.startswith("unanswerable") \
                or "not provided" in lowercased or "cannot be found" in lowercased:
            return "Unanswerable"

        return text.rstrip(".")


    def assemble_prompt(self, retrieved_chunks: list[str], question: str) -> tuple[str, str]:
        context_str = "\n\n".join(retrieved_chunks)

        system = (
            "You are an extractive QA system. Answer strictly from the provided context.\n"
            "1. If the context states the answer, set answerable=true and copy the exact "
            "span verbatim into answer. The answer must be the shortest span that answers "
            "the question — a few words, never a full sentence. No explanations, no extra words.\n"
            "2. If the context does not state the answer — even if it is on the same topic — "
            "set answerable=false and leave answer empty.\n"
            "3. Never use outside knowledge. Never guess."
        )

        prompt = (
            f"Context:\n{context_str}\n\n"
            f"Question: {question}\n\n"
            "Examples:\n"
            '{"answerable": true, "answer": "Berlin"}\n'
            '{"answerable": false, "answer": ""}\n\n'
            'Respond with JSON: {"answerable": true|false, "answer": "<exact span or empty>"}'
        )
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