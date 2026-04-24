"""
Document ingestion pipeline with production-grade error handling and GPU acceleration.

WHY: Ingestor must be fault-tolerant — one corrupted PDF shouldn't crash the entire process.
     Structured logging enables issue tracking in production.
"""

import os
import shutil
import logging
import json
from typing import List, Dict, Optional, Set
from pathlib import Path
from datetime import datetime

from llama_index.core import SimpleDirectoryReader, Document, VectorStoreIndex, StorageContext, Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.vector_stores.qdrant import QdrantVectorStore
import qdrant_client
from qdrant_client.models import Distance, VectorParams, PointStruct

from src.config import RAGConfig, get_model_provider, ModelProvider


# ============================================================================
# STRUCTURED LOGGING (Production Best Practice)
# ============================================================================
class StructuredLogger:
    """
    WHY: JSON logs are easy to parse in ELK/Datadog/Grafana.
         For employers, this shows production readiness.
    """

    def __init__(self, name: str):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)

        # Console handler
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter('%(message)s'))
        self.logger.addHandler(handler)

    def log(self, level: str, message: str, **kwargs):
        """Logs in JSON format for easy parsing."""
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": level,
            "message": message,
            **kwargs
        }
        self.logger.info(json.dumps(log_entry, ensure_ascii=False))


logger = StructuredLogger(__name__)


# ============================================================================
# CUSTOM EXCEPTIONS (Clean Code)
# ============================================================================
class IngestionError(Exception):
    """Base exception for ingestion pipeline."""
    pass


class DocumentLoadError(IngestionError):
    """Raised when document cannot be loaded."""
    pass


class VectorStoreError(IngestionError):
    """Raised when vector store operation fails."""
    pass


# ============================================================================
# DOCUMENT INGESTOR (Improved)
# ============================================================================
class DocumentIngestor:
    """
    Production-ready document ingestion pipeline with:
    - Comprehensive error handling
    - GPU-accelerated embeddings
    - Incremental ingestion (add new documents without recreating DB)
    - Batch processing
    - Detailed observability
    """

    def __init__(
            self,
            config: Optional[RAGConfig] = None,
            model_provider: Optional[ModelProvider] = None,
            existing_client: Optional[qdrant_client.QdrantClient] = None,
    ):
        """
        WHY: Dependency injection makes the class testable.
             existing_client lets the caller share an already-open Qdrant client,
             avoiding "already accessed" errors in Qdrant local mode.
        """
        self.config = config or RAGConfig()
        self.provider = model_provider or get_model_provider(self.config)
        self._existing_client = existing_client

        self.llm = self.provider.get_llm()
        self.embed_model = self.provider.get_embeddings()

        # Global LlamaIndex settings
        Settings.llm = self.llm
        Settings.embed_model = self.embed_model
        Settings.chunk_size = self.config.chunk_size
        Settings.chunk_overlap = self.config.chunk_overlap

        logger.log(
            "INFO",
            "DocumentIngestor initialized",
            chunk_size=self.config.chunk_size,
            device=self.config.embedding_device
        )

    def _validate_data_directory(self) -> None:
        """
        WHY: Fail fast — better to crash at startup than after 10 minutes of indexing.
        """
        data_path = Path(self.config.data_dir)

        if not data_path.exists():
            raise IngestionError(f"Data directory not found: {self.config.data_dir}")

        pdf_files = list(data_path.glob("*.pdf"))
        if not pdf_files:
            raise IngestionError(f"No PDF files found in {self.config.data_dir}")

        logger.log(
            "INFO",
            "Data directory validated",
            pdf_count=len(pdf_files),
            files=[f.name for f in pdf_files]
        )

    def _get_client(self) -> qdrant_client.QdrantClient:
        """Return the shared client if provided, else open a new one."""
        if self._existing_client is not None:
            return self._existing_client
        return qdrant_client.QdrantClient(path=self.config.db_path)

    def _get_indexed_documents(self) -> Set[str]:
        """
        Get list of already indexed document filenames from vector store.

        WHY: Enables incremental ingestion - skip already processed files.
        """
        try:
            client = self._get_client()

            # Check if collection exists
            collections = client.get_collections().collections
            if not any(c.name == self.config.collection_name for c in collections):
                return set()

            # Scroll through all points to get unique filenames
            indexed_files = set()
            offset = None

            while True:
                records, offset = client.scroll(
                    collection_name=self.config.collection_name,
                    limit=100,
                    offset=offset,
                    with_payload=True
                )

                for record in records:
                    if record.payload and "file_name" in record.payload:
                        indexed_files.add(record.payload["file_name"])

                if offset is None:
                    break

            return indexed_files

        except Exception as e:
            logger.log("WARNING", "Could not retrieve indexed documents", error=str(e))
            return set()

    def _load_documents(self, skip_existing: bool = False) -> List[Document]:
        """
        WHY: SimpleDirectoryReader requires llama-index-readers-file (not installed) to
             parse PDFs; without it the raw binary content is indexed instead of text.
             We use pypdf directly (already in requirements) for reliable text extraction.
             Try-catch per file — corrupted PDFs don't crash the entire process.
        """
        from pypdf import PdfReader as PyPDFReader

        documents = []
        errors = []

        data_path = Path(self.config.data_dir)
        pdf_files = list(data_path.glob("*.pdf"))

        # Get already indexed files if incremental mode
        indexed_files = self._get_indexed_documents() if skip_existing else set()

        if indexed_files:
            logger.log(
                "INFO",
                "Incremental mode: skipping already indexed files",
                indexed_count=len(indexed_files)
            )

        for pdf_file in pdf_files:
            # Skip if already indexed
            if skip_existing and pdf_file.name in indexed_files:
                logger.log(
                    "INFO",
                    "Skipping already indexed document",
                    file=pdf_file.name
                )
                continue

            try:
                pdf_reader = PyPDFReader(str(pdf_file))
                docs = []

                for page_num, page in enumerate(pdf_reader.pages):
                    text = page.extract_text() or ""
                    if not text.strip():
                        continue
                    doc = Document(
                        text=text,
                        metadata={
                            "file_name": pdf_file.name,
                            "file_path": str(pdf_file.resolve()),
                            "page_label": str(page_num + 1),
                            "file_type": "application/pdf",
                        }
                    )
                    docs.append(doc)

                documents.extend(docs)

                logger.log(
                    "INFO",
                    "Document loaded successfully",
                    file=pdf_file.name,
                    pages=len(docs)
                )

            except Exception as e:
                error_msg = f"Failed to load {pdf_file.name}: {str(e)}"
                errors.append(error_msg)
                logger.log(
                    "ERROR",
                    "Document load failed",
                    file=pdf_file.name,
                    error=str(e)
                )

        # Partial success is acceptable, but log warning
        if errors:
            logger.log(
                "WARNING",
                "Some documents failed to load",
                failed_count=len(errors),
                total_count=len(pdf_files)
            )

        if not documents:
            if skip_existing and errors == []:
                # All files were already indexed — valid incremental state
                return []
            raise DocumentLoadError("No documents were successfully loaded")

        return documents

    def _create_chunks(self, documents: List[Document]) -> List:
        """
        Dispatch to the configured chunking strategy.

        Strategies
        ----------
        sentence  SentenceSplitter — respects sentence boundaries (default).
                  Produces semantically coherent chunks.
        fixed     TokenTextSplitter — hard token-count boundaries.
                  Faster, but may cut mid-sentence. Useful as a baseline to
                  measure the benefit of sentence-aware splitting.
        """
        try:
            strategy = self.config.chunking_strategy
            if strategy == "fixed":
                from llama_index.core.node_parser import TokenTextSplitter
                splitter = TokenTextSplitter(
                    chunk_size=self.config.chunk_size,
                    chunk_overlap=self.config.chunk_overlap,
                )
            else:
                # "sentence" is the default and recommended strategy
                splitter = SentenceSplitter(
                    chunk_size=self.config.chunk_size,
                    chunk_overlap=self.config.chunk_overlap,
                )

            nodes = splitter.get_nodes_from_documents(documents)

            logger.log(
                "INFO",
                "Document chunking completed",
                strategy=strategy,
                total_nodes=len(nodes),
                avg_chunk_size=sum(len(n.text) for n in nodes) // len(nodes) if nodes else 0,
            )

            return nodes

        except Exception as e:
            raise IngestionError(f"Chunking failed: {str(e)}")

    def _initialize_vector_store(self, recreate: bool = False) -> qdrant_client.QdrantClient:
        """
        WHY: Explicit collection creation with parameters gives more control.
             Distance metric (Cosine vs Dot Product) is critical for search quality.
        """
        try:
            # Remove old database if recreating (only when no shared client)
            if recreate and self._existing_client is None and Path(self.config.db_path).exists():
                logger.log("INFO", "Removing old vector store", path=self.config.db_path)
                shutil.rmtree(self.config.db_path)

            client = self._get_client()

            # Check embedding dimensions
            # BGE-small-en-v1.5 = 384 dimensions
            test_embedding = self.embed_model.get_text_embedding("test")
            embedding_dim = len(test_embedding)

            # Create collection if it doesn't exist
            collections = client.get_collections().collections
            if not any(c.name == self.config.collection_name for c in collections):
                client.create_collection(
                    collection_name=self.config.collection_name,
                    vectors_config=VectorParams(
                        size=embedding_dim,
                        distance=Distance.COSINE
                    )
                )
                logger.log(
                    "INFO",
                    "Vector collection created",
                    collection=self.config.collection_name,
                    embedding_dim=embedding_dim
                )

            logger.log(
                "INFO",
                "Vector store initialized",
                embedding_dim=embedding_dim,
                distance_metric="Cosine"
            )

            return client

        except Exception as e:
            raise VectorStoreError(f"Vector store initialization failed: {str(e)}")

    def _create_or_update_index(
            self,
            nodes: List,
            client: qdrant_client.QdrantClient,
            is_update: bool = False
    ) -> VectorStoreIndex:
        """
        WHY: show_progress=True gives console feedback.
             Batch processing through embed_model is important for GPU efficiency.
        """
        try:
            vector_store = QdrantVectorStore(
                client=client,
                collection_name=self.config.collection_name
            )
            storage_context = StorageContext.from_defaults(vector_store=vector_store)

            if is_update:
                # Load existing index and add new nodes
                index = VectorStoreIndex.from_vector_store(vector_store)
                for node in nodes:
                    index.insert_nodes([node])
                logger.log(
                    "INFO",
                    "Vector index updated with new nodes",
                    new_nodes=len(nodes),
                    collection=self.config.collection_name
                )
            else:
                # Create new index from scratch
                index = VectorStoreIndex(
                    nodes,
                    storage_context=storage_context,
                    embed_model=self.embed_model,
                    show_progress=True
                )
                logger.log(
                    "INFO",
                    "Vector index created",
                    node_count=len(nodes),
                    collection=self.config.collection_name
                )

            return index

        except Exception as e:
            raise VectorStoreError(f"Index creation failed: {str(e)}")

    def ingest(self, recreate: bool = False) -> Dict[str, any]:
        """
        Main ingestion pipeline with comprehensive error handling.

        Args:
            recreate: If True, recreates the entire vector store.
                     If False, adds new documents incrementally.

        Returns:
            Dict with ingestion statistics and status.

        WHY: Returns structured result instead of None for observability.
        """
        start_time = datetime.utcnow()

        try:
            # Pipeline stages
            self._validate_data_directory()
            documents = self._load_documents(skip_existing=not recreate)

            if not documents:
                logger.log("INFO", "No new documents to ingest")
                return {
                    "status": "success",
                    "documents_loaded": 0,
                    "chunks_created": 0,
                    "duration_seconds": 0,
                    "mode": "incremental",
                    "message": "No new documents to ingest"
                }

            nodes = self._create_chunks(documents)
            client = self._initialize_vector_store(recreate=recreate)
            index = self._create_or_update_index(nodes, client, is_update=not recreate)

            duration = (datetime.utcnow() - start_time).total_seconds()

            result = {
                "status": "success",
                "documents_loaded": len(documents),
                "chunks_created": len(nodes),
                "duration_seconds": duration,
                "embedding_device": self.config.embedding_device,
                "mode": "full_recreate" if recreate else "incremental"
            }

            logger.log("INFO", "Ingestion completed successfully", **result)
            return result

        except IngestionError as e:
            logger.log("ERROR", "Ingestion failed", error=str(e), stage="pipeline")
            raise
        except Exception as e:
            logger.log("ERROR", "Unexpected error during ingestion", error=str(e))
            raise IngestionError(f"Ingestion failed: {str(e)}")

    def get_indexed_documents_list(self) -> List[str]:
        """
        Get list of all indexed document filenames.

        WHY: Needed for UI document selector.
        """
        return sorted(list(self._get_indexed_documents()))


# ============================================================================
# CLI INTERFACE
# ============================================================================
if __name__ == "__main__":
    import sys as _sys
    import os as _os
    # Allow running as `python src/ingestor.py` from project root
    _project_root = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
    if _project_root not in _sys.path:
        _sys.path.insert(0, _project_root)
    import argparse

    parser = argparse.ArgumentParser(description="Ingest documents into RAG system")
    parser.add_argument(
        "--data-dir",
        default="data",
        help="Directory containing PDF files"
    )
    parser.add_argument(
        "--db-path",
        default="db",
        help="Vector store database path"
    )
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="Recreate entire vector store instead of incremental update"
    )
    args = parser.parse_args()

    # Override config with CLI args
    config = RAGConfig(
        data_dir=args.data_dir,
        db_path=args.db_path
    )

    ingestor = DocumentIngestor(config=config)
    result = ingestor.ingest(recreate=args.recreate)

    print(f"\nIngestion completed in {result['duration_seconds']:.2f}s")
    print(f"   Documents: {result['documents_loaded']}")
    print(f"   Chunks: {result['chunks_created']}")
    print(f"   Device: {result['embedding_device']}")
    print(f"   Mode: {result['mode']}")