import os
import faiss
import numpy as np
import ollama
import random
from datasets import load_dataset
from sentence_transformers import SentenceTransformer




class NaiveRAG:
    def __init__(self, top_k: int = 3, dataset="rajpurkar/squad_v2", split="validation[:200]", model_name="llama3.2:3b"):
        self.top_k = top_k
        self.client = ollama.Client(host=os.getenv("OLLAMA_HOST", "http://ollama-service:11434"))
        self.embedder = SentenceTransformer("BAAI/bge-small-en-v1.5")
        self.dataset = dataset
        self.split = split
        self.unique_contexts, self.index = self.build_index()
        self.model_name = model_name

    def build_index(self):
        print("Loading SQuAD 2.0 validation split...")
        dataset = load_dataset(self.dataset, split=self.split)
        unique_contexts = list(dict.fromkeys(dataset["context"]))
        print(f"Extracted {len(unique_contexts)} unique context paragraphs.")

        print("Generating embeddings with BAAI/bge-small-en-v1.5...")
        context_embeddings = self.embedder.encode(unique_contexts, normalize_embeddings=True)

        dimension = context_embeddings.shape[1]
        index = faiss.IndexFlatIP(dimension)  # Inner product for normalized cosine similarity
        index.add(np.array(context_embeddings, dtype=np.float32))

        return unique_contexts, index

    def raw_generate(self, prompt: str) -> str:
        """Invokes raw generate endpoint with no chat template or system prompt."""
        response = self.client.generate(
            model=self.model_name,
            prompt=prompt,
            raw=True,
            options={"temperature": 0.0, "num_predict": 128}
        )
        return response["response"]

    def clean_completion(self, text: str) -> str:
        """Strips conversational boilerplate and isolates the concise target answer."""
        lines = [line.strip() for line in text.strip().split("\n") if line.strip()]
        if not lines:
            return ""
        text = lines[0]
        
        # Strip common model leading intros
        lowercased = text.lower()
        for prefix in ["answer:", "the answer is", "based on the context,"]:
            if lowercased.startswith(prefix):
                text = text[len(prefix):].strip()
                lowercased = text.lower()
                
        if lowercased.startswith("unanswerable") or "not provided" in lowercased or "cannot be found" in lowercased:
            return "Unanswerable"
            
        return text.rstrip(".")

    def run(self ,question: str, top_k: int = 3):
        # Retrieve top-k context chunks
        q_emb = self.embedder.encode([question], normalize_embeddings=True)
        _, indices = self.index.search(np.array(q_emb, dtype=np.float32), top_k)
        retrieved_chunks = [self.unique_contexts[idx] for idx in indices[0]]
        
        # Format raw prompt manually
        context_str = "\n\n".join(retrieved_chunks)
        # Updated Prompt Construction inside the evaluation loop:
        prompt = (
            f"Context:\n{context_str}\n\n"
            f"Instructions:\n"
            f"- Answer the question using ONLY an exact short phrase or word directly from the context.\n"
            f"- If the question cannot be answered using the context, respond with EXACTLY 'Unanswerable'.\n"
            f"- Do not write full sentences or explanations.\n\n"
            f"Question: {question}\n"
            f"Answer:"
        )

        raw_output = self.raw_generate(prompt)
        answer = self.clean_completion(raw_output)

        return prompt, answer, retrieved_chunks
