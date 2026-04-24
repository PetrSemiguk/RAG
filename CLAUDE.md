# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

### Setup
```bash
python -m venv .venv
source .venv/Scripts/activate  # Windows: .venv\Scripts\activate.bat
pip install -r requirements.txt
```

### Ingest Documents
```bash
# First-time or rebuild index from scratch
python src/ingestor.py --data-dir data --db-path db --recreate

# Incremental (skip already-indexed files)
python src/ingestor.py --data-dir data --db-path db
```

### Run the App
```bash
streamlit run app.py
# Opens at http://localhost:8501
```

### Prerequisites
- **LM Studio** must be running locally at `http://localhost:1234/v1` with a model loaded (default LLM provider)
- Place PDF files in `data/` before ingestion
- Optional: Set `COHERE_API_KEY` env var to enable reranking

## Architecture

### Component Overview

```
app.py (Streamlit UI)
  └── RAGQueryEngine (src/engine.py)
        ├── Retrieval: HybridSearchStrategy or VectorOnlyStrategy
        │     ├── Qdrant VectorRetriever (384-dim cosine, db/ path)
        │     └── BM25Retriever (keyword, in-memory docstore)
        │     └── QueryFusionRetriever (reciprocal rank fusion)
        ├── Post-processors: SimilarityPostprocessor (≥0.6) + optional CohereRerank
        ├── AdaptiveContextManager (prevents >2048-token context overflow)
        └── LLM via ModelProvider (src/config.py)
              ├── LocalModelProvider → LM Studio @ localhost:1234
              └── OpenAIModelProvider → gpt-3.5-turbo

DocumentIngestor (src/ingestor.py)
  └── PDF → SentenceSplitter (512-char chunks, 50-char overlap)
        └── HuggingFace embeddings (BAAI/bge-small-en-v1.5, GPU-aware)
              └── Qdrant VectorStore (collection: "advanced_rag", db/)
```

### Key Design Patterns
- **Strategy pattern** for retrieval (`VectorOnlyStrategy` / `HybridSearchStrategy`) — swap without touching engine logic
- **Factory pattern** for model providers (`ModelProvider` ABC in `config.py`) — swap LLM/embedding providers without touching engine logic
- **Structured query result**: `engine.query()` always returns `{"answer": str, "sources": List[Dict], "metadata": Dict}` — UI and callers depend on this contract

### Configuration (`src/config.py`)
All tunables live in `RAGConfig` (Pydantic model):
- `llm_provider`: `"local"` (LM Studio) or `"openai"`
- `vector_top_k` / `bm25_top_k` (default 5 each), `rerank_top_n` (default 3)
- `chunk_size` / `chunk_overlap` (default 512 / 50)
- `similarity_threshold` (default 0.6)
- `SYSTEM_PROMPT_EN`: enforces answers strictly from retrieved context

### Document Filtering
Qdrant metadata field `file_name` is attached to every chunk at ingestion. `engine.update_document_filter(filename)` dynamically restricts retrieval to a single PDF without reloading the engine. `get_available_documents()` lists all indexed PDFs.

### Ingestion Modes
- `recreate=True`: drops and rebuilds the Qdrant collection
- `skip_existing=True` (default): skips files already in the index for incremental updates
- Per-file error handling — corrupted PDFs are logged and skipped, not fatal

### GPU Acceleration
Embedding uses `batch_size=32` on CUDA, `8` on CPU. Device is auto-detected in `ingestor.py` and `config.py`.
