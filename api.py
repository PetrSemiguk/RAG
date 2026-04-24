"""
REST API layer for the RAG system.

Exposes RAGQueryEngine over HTTP so any frontend, CI script, or external
service can query the system — not just the Streamlit UI.

Run:
    uvicorn api:app --reload          # development
    uvicorn api:app --host 0.0.0.0    # production / Docker

Endpoints:
    GET  /health      — vector store + embeddings status
    GET  /documents   — list all indexed PDFs
    POST /query       — ask a question, get an answer with sources
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

app = FastAPI(
    title="Advanced RAG API",
    description="Hybrid BM25 + Vector retrieval with local or cloud LLM.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# Lazy singletons — loaded on the first request to avoid blocking startup
_engine = None
_qlogger = None


def _get_engine():
    global _engine, _qlogger
    if _engine is None:
        from src.config import RAGConfig
        from src.engine import RAGQueryEngine
        from src.observability.query_logger import QueryLogger

        config = RAGConfig.from_yaml("config.yaml")
        _engine = RAGQueryEngine(config=config, use_hybrid=True, use_reranking=False)
        _qlogger = QueryLogger(db_path=config.sqlite_path)
    return _engine, _qlogger


# ── Schemas ────────────────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, description="The user question")
    document_filter: Optional[List[str]] = Field(
        default=None,
        description="Restrict retrieval to these PDF filenames (None = all documents)",
    )
    history: Optional[List[Dict[str, str]]] = Field(
        default=None,
        description="Prior conversation turns, each as {role: str, content: str}",
    )

    model_config = {"json_schema_extra": {
        "example": {
            "question": "What is the main contribution of this paper?",
            "document_filter": ["attention_is_all_you_need.pdf"],
            "history": [],
        }
    }}


class QueryResponse(BaseModel):
    answer: str
    sources: List[Dict[str, Any]]
    metadata: Dict[str, Any]


class DocumentsResponse(BaseModel):
    documents: List[str]
    count: int


# ── Endpoints ──────────────────────────────────────────────────────────────────

@app.get(
    "/health",
    summary="Health check",
    response_description="Status of vector store, embeddings, and LLM",
)
def health() -> Dict[str, Any]:
    """Returns live status of all system components."""
    try:
        engine, _ = _get_engine()
        return engine.health_check()
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"System not ready: {exc}",
        )


@app.get(
    "/documents",
    response_model=DocumentsResponse,
    summary="List indexed documents",
)
def list_documents() -> DocumentsResponse:
    """Return all PDF filenames currently stored in the vector index."""
    try:
        engine, _ = _get_engine()
        docs = engine.get_available_documents()
        return DocumentsResponse(documents=docs, count=len(docs))
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        )


@app.post(
    "/query",
    response_model=QueryResponse,
    summary="Ask a question",
    response_description="Generated answer with source citations and metadata",
)
def query(request: QueryRequest) -> QueryResponse:
    """
    Retrieve the most relevant document chunks and synthesise an answer.

    - Use `document_filter` to scope the search to specific PDFs.
    - Pass `history` to give the model conversation context.
    - The answer is grounded strictly in retrieved context (no hallucination).
    """
    try:
        engine, qlogger = _get_engine()
        if request.document_filter is not None:
            engine.update_document_filter(request.document_filter or None)
        result = engine.query(request.question, conversation_history=request.history)
        if qlogger:
            qlogger.log_query(request.question, result)
        return QueryResponse(**result)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
