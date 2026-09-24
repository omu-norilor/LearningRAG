# Copilot instructions for LearningRAG

- This repo is a minimal RAG evaluation harness for SQuAD v2. The runtime is containerized under `naive/infra`, and the actual logic lives in `naive/code/src`.
- `naive/code/src/rag.py` is the canonical implementation. `NaiveRAG.__init__()` loads the `BAAI/bge-small-en-v1.5` embedder, builds a deduplicated FAISS corpus from unique `context` paragraphs, and defaults to `OLLAMA_HOST=http://ollama-service:11434`.
- Retrieval is cosine-style similarity via normalized embeddings: `embedder.encode(..., normalize_embeddings=True)` followed by `faiss.IndexFlatIP`. The indexing logic assumes unit-normalized vectors; do not change retrieval to raw dot products without normalizing inputs.
- `NaiveRAG.run()` retrieves `top_k` chunks, constructs a prompt that forces a short exact answer from the supplied context, calls `ollama.Client.generate(..., raw=True)`, and then applies `clean_completion()` to strip boilerplate and map impossible answers to `Unanswerable`.
- `naive/code/src/eval.py` is the benchmark driver. It loads `rajpurkar/squad_v2`, deduplicates contexts, computes retrieval metrics (`Recall@k`, `MRR`), runs the same raw Ollama prompt path, and evaluates with `evaluate.load("squad_v2")` before writing `/app/data/failure_analysis.json`.
- `naive/code/src/test.py` is the smoke-test entry point: it samples a random validation example, calls `NaiveRAG.run()`, and prints the prompt plus completion to validate the full RAG path without running the full evaluation set.
- `naive/code/src/bootstrap_data.py` is the setup script. It preloads the SQuAD dataset, the `squad_v2` metric, the embedding model, and the Llama GGUF file into `/app/models` before a run; use it when caches are missing.
- The project expects Docker Compose services from `naive/infra/docker-compose.yaml`: `rag-dev` runs the Python app, and `ollama` exposes the model endpoint on port `11435` (host) / `11434` (container).
- The app mounts `../code/src` to `/app/src`, `../data` to `/app/data`, and `../models` to `/app/models`, so prefer editing the source under `naive/code/src` and treating the container volumes as runtime state.
- For reproducible local runs, use the Compose workflow instead of running Python directly on the host: `docker compose -f naive/infra/docker-compose.yaml up --build -d`, then `docker compose -f naive/infra/docker-compose.yaml exec rag-dev python /app/src/test.py` or `... python /app/src/eval.py`.
- Keep prompt wording stable when editing generation behavior. The current evaluation depends on exact-match logic and the strict instruction: “answer using only an exact short phrase or word directly from the context” and “respond with EXACTLY 'Unanswerable'”.
- Because `eval.py` defines failure cases as `Retrieval Miss` vs `Generation Miss`, any prompt or retrieval change should be checked with the failure log to separate retrieval quality from answer-quality regressions.
- The repo is intentionally simple and script-oriented: prefer focused changes in the existing files over introducing new project scaffolding or app frameworks.
- Do not assume a standard web app or package layout; this is a research/ML pipeline with dataset/model downloads and containerized execution.
- When adding features, keep the data flow consistent: dataset -> unique context corpus -> FAISS index -> LLM prompt -> cleaned answer -> evaluation metrics.
- Use `raw=True` generation for this project; the code intentionally avoids chat templating and instead manually formats the instructions.
- If you need to inspect runtime behavior, start from `rag.py` and then compare with `eval.py`; those two files define the project’s retrieval + generation contract.
