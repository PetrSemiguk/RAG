"""
Configuration module with Pydantic settings validation and GPU support.

WHY: Pydantic ensures type safety and validation at application startup,
     not at runtime. GPU acceleration is critical for production-grade systems.
     from_yaml() loads all tunables from config.yaml so there are no magic
     numbers scattered through the codebase.
"""

import os
from typing import Literal, Optional
from abc import ABC, abstractmethod
from pydantic import BaseModel, Field, field_validator, ConfigDict
from dotenv import load_dotenv
import torch

from llama_index.llms.openai import OpenAI
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.llms.llm import LLM
from llama_index.core.base.embeddings.base import BaseEmbedding

load_dotenv()


# ============================================================================
# PROJECT SETTINGS (Pydantic for validation)
# ============================================================================
class RAGConfig(BaseModel):
    """
    All tunables in one place. Pydantic validates types and value ranges
    before the application starts. Load from config.yaml via from_yaml().
    """
    # LLM Settings
    use_local_llm: bool = True
    lm_studio_base_url: str = "http://localhost:1234/v1"
    lm_studio_api_key: str = "lm-studio"
    openai_model: str = "gpt-3.5-turbo"
    llm_temperature: float = Field(default=0.1, ge=0.0, le=1.0)

    # Embedding Settings
    embedding_model: str = "BAAI/bge-base-en-v1.5"
    embedding_device: str = Field(
        default="cuda" if torch.cuda.is_available() else "cpu"
    )

    # Chunking Settings
    chunk_size: int = Field(default=512, ge=128, le=2048)
    chunk_overlap: int = Field(default=50, ge=0, le=200)
    chunking_strategy: Literal["sentence", "semantic"] = "sentence"
    semantic_breakpoint_percentile: int = Field(default=95, ge=50, le=99)

    # Retrieval Settings
    vector_top_k: int = Field(default=8, ge=1, le=20)
    bm25_top_k: int = Field(default=5, ge=1, le=20)
    rerank_top_n: int = Field(default=3, ge=1, le=10)

    # Reranking Settings
    use_cross_encoder: bool = False
    cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # System Paths
    data_dir: str = "data"
    db_path: str = "db"
    collection_name: str = "advanced_rag"
    results_dir: str = "results"
    logs_dir: str = "logs"

    # Observability
    sqlite_path: str = "logs/queries.db"

    @field_validator("chunk_overlap")
    @classmethod
    def validate_overlap(cls, v: int, info) -> int:
        chunk_size = info.data.get("chunk_size", 512)
        if v >= chunk_size:
            raise ValueError(f"chunk_overlap ({v}) must be < chunk_size ({chunk_size})")
        return v

    model_config = ConfigDict(frozen=True)

    @classmethod
    def from_yaml(cls, yaml_path: str = "config.yaml") -> "RAGConfig":
        """
        Load configuration from a YAML file, falling back to defaults for
        any missing keys. This is the recommended constructor.

        WHY: Keeps all tunables in one human-readable file rather than
             scattered across source files or environment variables.
        """
        from src.utils import load_yaml_config
        data = load_yaml_config(yaml_path)
        if not data:
            return cls()

        llm = data.get("llm", {})
        emb = data.get("embedding", {})
        chunking = data.get("chunking", {})
        retrieval = data.get("retrieval", {})
        reranking = data.get("reranking", {})
        paths = data.get("paths", {})
        obs = data.get("observability", {})

        device = emb.get("device", "auto")
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        return cls(
            use_local_llm=llm.get("use_local", True),
            lm_studio_base_url=llm.get("lm_studio_url", "http://localhost:1234/v1"),
            lm_studio_api_key=llm.get("lm_studio_api_key", "lm-studio"),
            llm_temperature=llm.get("temperature", 0.1),
            openai_model=llm.get("openai_model", "gpt-3.5-turbo"),
            embedding_model=emb.get("model", "BAAI/bge-base-en-v1.5"),
            embedding_device=device,
            chunk_size=chunking.get("chunk_size", 512),
            chunk_overlap=chunking.get("chunk_overlap", 50),
            chunking_strategy=chunking.get("strategy", "sentence"),
            semantic_breakpoint_percentile=chunking.get("semantic_breakpoint_percentile", 95),
            vector_top_k=retrieval.get("vector_top_k", 8),
            bm25_top_k=retrieval.get("bm25_top_k", 5),
            rerank_top_n=retrieval.get("rerank_top_n", 3),
            use_cross_encoder=reranking.get("enabled", False),
            cross_encoder_model=reranking.get(
                "cross_encoder_model", "cross-encoder/ms-marco-MiniLM-L-6-v2"
            ),
            data_dir=paths.get("data_dir", "data"),
            db_path=paths.get("db_path", "db"),
            collection_name=paths.get("collection_name", "advanced_rag"),
            results_dir=paths.get("results_dir", "results"),
            logs_dir=paths.get("logs_dir", "logs"),
            sqlite_path=obs.get("sqlite_path", "logs/queries.db"),
        )


# ============================================================================
# ABSTRACT BASE CLASSES (for extensibility)
# ============================================================================
class ModelProvider(ABC):
    """
    Abstraction layer for LLM + embedding providers.

    WHY: Swapping OpenAI → Anthropic → Local requires changing only
         one subclass, not the engine or ingestor.
    """

    @abstractmethod
    def get_llm(self) -> LLM:
        """Returns configured LLM instance."""
        pass

    @abstractmethod
    def get_embeddings(self) -> BaseEmbedding:
        """Returns configured embedding model."""
        pass


class LocalModelProvider(ModelProvider):
    """LM Studio provider with GPU-accelerated HuggingFace embeddings."""

    def __init__(self, config: RAGConfig):
        self.config = config

    def get_llm(self) -> LLM:
        """
        WHY: OpenAI-compatible API lets us reuse llama-index's OpenAI
             integration with any local model served by LM Studio.
        """
        return OpenAI(
            api_key=self.config.lm_studio_api_key,
            api_base=self.config.lm_studio_base_url,
            temperature=self.config.llm_temperature,
        )

    def get_embeddings(self) -> BaseEmbedding:
        """
        WHY: device="cuda" forces HuggingFace onto the GPU —
             10-15x speedup vs CPU for RTX-class cards.
        """
        return HuggingFaceEmbedding(
            model_name=self.config.embedding_model,
            device=self.config.embedding_device,
            embed_batch_size=32 if self.config.embedding_device == "cuda" else 8,
        )


class OpenAIModelProvider(ModelProvider):
    """Production OpenAI provider."""

    def __init__(self, config: RAGConfig):
        self.config = config
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY not found in environment")

    def get_llm(self) -> LLM:
        return OpenAI(
            model=self.config.openai_model,
            api_key=self.api_key,
            temperature=self.config.llm_temperature,
        )

    def get_embeddings(self) -> BaseEmbedding:
        from llama_index.embeddings.openai import OpenAIEmbedding
        return OpenAIEmbedding(api_key=self.api_key)


# ============================================================================
# FACTORY PATTERN
# ============================================================================
def get_model_provider(config: RAGConfig) -> ModelProvider:
    """
    WHY: Factory pattern hides object creation details.
         Adding a new provider = adding 1 class, 1 branch — no other changes.
    """
    if config.use_local_llm:
        return LocalModelProvider(config)
    return OpenAIModelProvider(config)


# ============================================================================
# SYSTEM PROMPTS
# ============================================================================
SYSTEM_PROMPT_EN = """You are a professional AI assistant for technical document analysis.

CRITICAL RULES:
1. Answer ONLY based on the provided context from documents.
2. If the context does NOT contain the answer → honestly say: "This information is not present in the provided documents."
3. Do NOT invent facts, dates, numbers, or names — only what is explicitly written in the context.
4. Answer in a structured and concise manner.
5. If the question is ambiguous → ask for clarification.

EXTRACTION RULES (highest priority):
6. If the retrieved chunks contain a numbered or bulleted list, reproduce it FULLY and EXACTLY — do not paraphrase, collapse, or omit any items.
7. If concrete details are present (names, numbers, dates, steps, items), enumerate every one of them explicitly. NEVER replace specific details with vague summaries like "several options" or "various methods".
8. When a chunk contains multiple distinct facts, present each fact separately — do not merge them into a single generic sentence.

RESPONSE FORMAT:
- Direct answer to the question.
- If the source contains a list → reproduce the full list verbatim, preserving original numbering/bullets.
- Additional specific details extracted from the context (values, names, steps).
- Reference to source: "According to document [name]..."

EXAMPLE:
Question: "What retrieval strategies are supported?"
Context: "...The system supports three strategies: 1) Vector search, 2) BM25 keyword search, 3) Hybrid search with reciprocal rank fusion..."
Answer: "The following retrieval strategies are supported:
1) Vector search
2) BM25 keyword search
3) Hybrid search with reciprocal rank fusion
According to document [name]."
"""


# ============================================================================
# USAGE EXAMPLE
# ============================================================================
if __name__ == "__main__":
    config = RAGConfig.from_yaml("config.yaml")
    print(f"Device: {config.embedding_device}")
    print(f"Chunking: {config.chunking_strategy} (size={config.chunk_size})")
    print(f"Retrieval: vector_top_k={config.vector_top_k}, bm25_top_k={config.bm25_top_k}")
