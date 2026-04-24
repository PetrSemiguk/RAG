"""
Tests for src/config.py.

Requires the full ML stack (torch + llama-index).
Skipped automatically in CI (requirements-dev.txt only).
Run locally after: pip install -r requirements.txt
"""
import pytest

# Skip this entire module if ML stack is not installed
pytest.importorskip("torch", reason="requires torch — run: pip install -r requirements.txt")
pytest.importorskip("llama_index", reason="requires llama-index — run: pip install -r requirements.txt")

from pydantic import ValidationError

from src.config import RAGConfig


class TestRAGConfigDefaults:
    def test_chunk_size(self):
        assert RAGConfig().chunk_size == 512

    def test_chunk_overlap(self):
        assert RAGConfig().chunk_overlap == 50

    def test_top_k(self):
        cfg = RAGConfig()
        assert cfg.vector_top_k == 5
        assert cfg.bm25_top_k == 5

    def test_rerank_top_n(self):
        assert RAGConfig().rerank_top_n == 3

    def test_chunking_strategy(self):
        assert RAGConfig().chunking_strategy == "sentence"

    def test_embedding_device_is_valid(self):
        assert RAGConfig().embedding_device in ("cpu", "cuda")


class TestRAGConfigValidation:
    def test_overlap_must_be_less_than_chunk_size(self):
        with pytest.raises(ValidationError):
            RAGConfig(chunk_size=256, chunk_overlap=256)

    def test_valid_custom_chunk_params(self):
        cfg = RAGConfig(chunk_size=1024, chunk_overlap=100)
        assert cfg.chunk_size == 1024
        assert cfg.chunk_overlap == 100

    def test_temperature_upper_bound(self):
        with pytest.raises(ValidationError):
            RAGConfig(llm_temperature=1.5)

    def test_temperature_lower_bound(self):
        with pytest.raises(ValidationError):
            RAGConfig(llm_temperature=-0.1)

    def test_config_is_immutable(self):
        cfg = RAGConfig()
        with pytest.raises(Exception):
            cfg.chunk_size = 999  # type: ignore[misc]


class TestRAGConfigFromYaml:
    def test_missing_file_falls_back_to_defaults(self):
        cfg = RAGConfig.from_yaml("__nonexistent_config__.yaml")
        assert isinstance(cfg, RAGConfig)
        assert cfg.chunk_size == 512

    def test_parses_chunking_section(self, tmp_path):
        f = tmp_path / "config.yaml"
        f.write_text(
            "chunking:\n  chunk_size: 256\n  chunk_overlap: 30\n"
            "retrieval:\n  vector_top_k: 3\n"
        )
        cfg = RAGConfig.from_yaml(str(f))
        assert cfg.chunk_size == 256
        assert cfg.vector_top_k == 3

    def test_empty_yaml_returns_defaults(self, tmp_path):
        f = tmp_path / "empty.yaml"
        f.write_text("")
        cfg = RAGConfig.from_yaml(str(f))
        assert cfg.chunk_size == 512
