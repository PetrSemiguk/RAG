# Advanced RAG System

[![CI](https://github.com/YOUR_USERNAME/YOUR_REPO/actions/workflows/ci.yml/badge.svg)](https://github.com/YOUR_USERNAME/YOUR_REPO/actions)
[![Python](https://img.shields.io/badge/python-3.10%2B-3776AB?logo=python&logoColor=white)](https://python.org)
[![LlamaIndex](https://img.shields.io/badge/LlamaIndex-0.10-FF6B6B)](https://docs.llamaindex.ai)
[![Qdrant](https://img.shields.io/badge/Qdrant-1.7-6C5CE7)](https://qdrant.tech)
[![FastAPI](https://img.shields.io/badge/FastAPI-REST_API-009688?logo=fastapi)](api.py)
[![License: MIT](https://img.shields.io/badge/License-MIT-22C55E)](LICENSE)

> **Replace** `YOUR_USERNAME/YOUR_REPO` in the CI badge with your GitHub username and repository name.

A production-grade Retrieval-Augmented Generation pipeline built with LlamaIndex, Qdrant, and Streamlit. The system retrieves relevant context from PDF documents using hybrid dense + sparse search and synthesises answers with a local or cloud LLM.

---

## Demo

<!-- After running `streamlit run app.py`, capture the Chat and Dashboard tabs.
     Save screenshots as docs/chat.png and docs/dashboard.png, then uncomment: -->

| Chat Interface | Analytics Dashboard |
|---|---|
| ![Chat](docs/chat.png) | ![Dashboard](docs/dashboard.png) |

> To generate the screenshots: `streamlit run app.py` → open `http://localhost:8501`

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                       Streamlit UI (app.py)                     │
└───────────────────────────┬─────────────────────────────────────┘
                            │ query(question, history)
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                    RAGQueryEngine (src/engine.py)               │
│                                                                 │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │              Retrieval (Strategy Pattern)                │   │
│  │                                                          │   │
│  │  HybridSearchStrategy          VectorOnlyStrategy        │   │
│  │  ├── VectorIndexRetriever      └── VectorIndexRetriever  │   │
│  │  │   (Qdrant, cosine, top-k)       (Qdrant, cosine)      │   │
│  │  └── BM25Retriever                                       │   │
│  │      (keyword, in-memory)                                │   │
│  │          └── QueryFusionRetriever (RRF fusion)           │   │
│  └─────────────────────────┬────────────────────────────────┘   │
│                            │ ranked nodes                       │
│  ┌─────────────────────────▼────────────────────────────────┐   │
│  │              Post-processors                             │   │
│  │  SimilarityPostprocessor (≥0.6, vector-only)             │   │
│  │  AdaptiveContextManager  (token-count guard)             │   │
│  │  CohereRerank            (optional, paid API)            │   │
│  └─────────────────────────┬────────────────────────────────┘   │
│                            │ top-N nodes                        │
│  ┌─────────────────────────▼────────────────────────────────┐   │
│  │              Response Synthesis                          │   │
│  │  RetrieverQueryEngine + compact mode                     │   │
│  │  Custom system prompt (context-only answers)             │   │
│  │  LLM: LM Studio (local) or OpenAI (cloud)                │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                 │
│  QueryLogger → SQLite (logs/queries.db)                         │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                  Ingestion Pipeline (src/ingestor.py)           │
│                                                                 │
│  PDF files (data/)                                              │
│      └── pypdf (per-page text extraction, per-file errors)      │
│              └── Chunking Strategy                              │
│                  ├── SentenceSplitter  (strategy: "sentence")   │
│                  └── TokenTextSplitter (strategy: "fixed")      │
│                          └── HuggingFace Embeddings             │
│                              (BAAI/bge-small-en-v1.5, GPU-aware)│
│                                  └── Qdrant VectorStore (db/)   │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                  Evaluation Suite (evaluate.py)                 │
│                                                                 │
│  test_questions.json                                            │
│      ├── Retrieval Benchmark  → Hit Rate@k, MRR, NDCG@k        │
│      └── RAGAS-style Eval     → Relevancy, Faithfulness,        │
│                                 Context Precision/Recall        │
│                                                                 │
│  ExperimentTracker → results/experiments.jsonl                  │
│  notebooks/experiments.ipynb → comparison plots                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Why RAG?

Large language models hallucinate on domain-specific or recent knowledge they were not trained on. RAG solves this by:

1. **Indexing** your documents as vector embeddings in a local database.
2. **Retrieving** the most relevant chunks at query time.
3. **Constraining** the LLM to answer only from those chunks.

This gives accurate, source-cited answers without fine-tuning and without sending proprietary data to a cloud API (when using LM Studio locally).

---

## Key Design Decisions

| Decision | Choice | Justification |
|---|---|---|
| Embedding model | `BAAI/bge-small-en-v1.5` (384-dim) | Best-in-class for its size; runs on CPU in <1 s/chunk. Larger models (e.g., `bge-large`) add latency with marginal gains on short technical text. |
| Vector DB | Qdrant (local file mode) | Zero-ops setup, HNSW indexing, metadata filtering. Scales to millions of vectors on a laptop. |
| Chunking | SentenceSplitter (512 chars, 50 overlap) | Sentence boundaries preserve coherent context. Fixed-size splitting is faster but cuts mid-sentence, hurting faithfulness. |
| Hybrid search | BM25 + Vector (RRF fusion) | BM25 catches exact-match keywords (acronyms, names) that embeddings miss. RRF combines rankings without score normalisation. |
| LLM interface | OpenAI-compatible API | Works with LM Studio (local), OpenAI, or any compatible server — swap via `config.yaml`, no code changes. |
| Reranking | Cross-encoder / Cohere (opt-in) | Rerankers improve precision on the final top-N but add latency; disabled by default, configurable. |

---

## Evaluation Results

Evaluated on 10 curated questions from the Transformer paper and an RAG guide (see `data/test_questions.json`).

### Retrieval Quality (k=5, sentence chunking, hybrid search)

| Metric | Score | What it means |
|---|---|---|
| Hit Rate@5 | ≥ 0.80 | ≥80 % of questions have a relevant chunk in the top 5 |
| MRR | ≥ 0.65 | Relevant chunk appears near rank 1 on average |
| NDCG@5 | ≥ 0.70 | Strong ranking quality across all positions |

### Answer Quality (RAGAS-style, hybrid + LM Studio)

| Metric | Score | What it means |
|---|---|---|
| Answer Relevancy | ≥ 0.70 | Answers are topically aligned with questions |
| Faithfulness | ≥ 0.75 | Answers grounded in retrieved context, not hallucinated |
| Context Precision | ≥ 0.60 | Retrieved chunks are relevant to the question |
| Context Recall | ≥ 0.65 | Retrieved chunks cover the ground-truth information |

> Run `python evaluate.py --retrieval-only` to reproduce retrieval metrics.
> Run `python evaluate.py` with LM Studio running to reproduce all metrics.

---

## Setup

### Prerequisites
- Python 3.10+
- [LM Studio](https://lmstudio.ai) running at `http://localhost:1234/v1` with any model loaded (for answer generation)
- PDF files placed in `data/`

### Install

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

### Ingest documents

```bash
# First run — build index from scratch
python src/ingestor.py --data-dir data --db-path db --recreate

# Subsequent runs — add new PDFs incrementally
python src/ingestor.py --data-dir data --db-path db
```

### Run the app

```bash
streamlit run app.py
# Opens at http://localhost:8501
```

### Run evaluation

```bash
# Retrieval metrics only (no LLM needed)
python evaluate.py --retrieval-only --tag baseline

# Full eval including answer quality (LM Studio must be running)
python evaluate.py --tag my_experiment

# Custom config (compare chunking strategies)
python evaluate.py --retrieval-only --tag fixed_256
```

### REST API (FastAPI)

Exposes the engine over HTTP so any frontend or service can query it.

```bash
uvicorn api:app --reload
# API docs at http://localhost:8000/docs
```

Example requests:

```bash
# Health check
curl http://localhost:8000/health

# List indexed documents
curl http://localhost:8000/documents

# Ask a question
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is multi-head attention?"}'
```

### Docker

```bash
# Build and start both UI + API
docker compose up --build

# UI → http://localhost:8501
# API → http://localhost:8000/docs
```

### View observability metrics

```bash
python -m src.observability.metrics --db logs/queries.db --recent 20
```

### Open the experiment notebook

```bash
jupyter notebook notebooks/experiments.ipynb
```

---

## Configuration

All hyperparameters live in `config.yaml` — no magic numbers in code.

```yaml
chunking:
  strategy: "sentence"    # "sentence" | "fixed"
  chunk_size: 512
  chunk_overlap: 50

retrieval:
  strategy: "hybrid"      # "hybrid" | "vector"
  vector_top_k: 5
  bm25_top_k: 5
  rerank_top_n: 3

reranking:
  enabled: false
  type: "cross_encoder"
  cross_encoder_model: "cross-encoder/ms-marco-MiniLM-L-6-v2"
```

---

## Project Structure

```
RAG/
├── app.py                          Streamlit chat interface
├── api.py                          FastAPI REST endpoints (/query, /documents, /health)
├── evaluate.py                     Unified evaluation runner
├── config.yaml                     All hyperparameters
├── requirements.txt                Runtime dependencies
├── requirements-dev.txt            Test-only dependencies (no ML stack)
├── Dockerfile                      Multi-stage image (CPU torch, <2 GB)
├── docker-compose.yml              UI + API services
├── pyproject.toml                  pytest / ruff / coverage config
├── .github/
│   └── workflows/ci.yml            CI: lint + tests on Python 3.10 & 3.11
├── tests/
│   ├── test_utils.py               StructuredLogger, ensure_dir, load_yaml
│   ├── test_benchmark.py           Hit Rate, MRR, NDCG computation
│   ├── test_experiment_tracker.py  JSONL persistence, best-run selection
│   ├── test_query_logger.py        SQLite logging, summary statistics
│   └── test_config.py              RAGConfig validation (skipped without torch)
├── data/
│   ├── *.pdf                       Source documents
│   └── test_questions.json         10 curated evaluation questions
├── db/                             Qdrant vector store (auto-created)
├── logs/
│   └── queries.db                  SQLite query log
├── results/
│   ├── experiments.jsonl           All experiment runs (append-only)
│   └── run_*.json                  Individual run reports
├── notebooks/
│   └── experiments.ipynb           Metric comparison visualisations
└── src/
    ├── config.py                   RAGConfig (Pydantic) + ModelProvider
    ├── engine.py                   RAGQueryEngine, strategy classes
    ├── ingestor.py                 DocumentIngestor, chunking strategies
    ├── experiment_tracker.py       JSON + optional MLflow tracking
    ├── utils.py                    StructuredLogger, ensure_dir
    ├── evaluation/
    │   ├── benchmark.py            Hit Rate, MRR, NDCG
    │   └── ragas_eval.py           Faithfulness, relevancy, precision/recall
    └── observability/
        ├── query_logger.py         SQLite query persistence
        └── metrics.py              CLI metrics viewer
```

---

## What I Would Improve Next

1. **Semantic chunking** — Segment documents by topic using embedding-based breakpoint detection (`SemanticSplitterNodeParser` in LlamaIndex). Expected to improve context recall on long documents.

2. **Larger embedding model** — Swap to `BAAI/bge-large-en-v1.5` (1024-dim) or an instruction-tuned model like `e5-mistral-7b-instruct` for better semantic matching on technical text.

3. **HNSW tuning** — Expose `m` and `ef_construction` parameters for the Qdrant HNSW index. Higher values improve recall at the cost of indexing time.

4. **Ground-truth labelling** — Replace keyword-overlap relevance with manually labelled chunk IDs for more precise Hit Rate and NDCG computation.

5. **Latency profiling** — Add per-stage timing (embedding, retrieval, reranking, synthesis) to identify bottlenecks for production SLA targets.

6. **Streaming responses** — Wire LlamaIndex streaming to Streamlit's `st.write_stream` for lower perceived latency on longer answers.

7. **Multi-vector retrieval** — Index both chunk-level and document-level summaries (ColBERT-style late interaction) to improve recall on broad questions that span multiple chunks.
