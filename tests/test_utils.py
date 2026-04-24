"""Tests for src/utils.py — no ML dependencies required."""
import json
import logging
from pathlib import Path

import pytest

from src.utils import StructuredLogger, ensure_dir, load_yaml_config


class TestEnsureDir:
    def test_creates_nested_dirs(self, tmp_path):
        target = tmp_path / "a" / "b" / "c"
        result = ensure_dir(target)
        assert target.is_dir()
        assert result == target

    def test_idempotent_on_existing_dir(self, tmp_path):
        ensure_dir(tmp_path)
        ensure_dir(tmp_path)  # must not raise
        assert tmp_path.is_dir()

    def test_accepts_string_path(self, tmp_path):
        target = str(tmp_path / "new_dir")
        ensure_dir(target)
        assert Path(target).is_dir()

    def test_returns_path_object(self, tmp_path):
        result = ensure_dir(tmp_path / "x")
        assert isinstance(result, Path)


class TestLoadYamlConfig:
    def test_missing_file_returns_empty_dict(self):
        assert load_yaml_config("__definitely_not_a_file__.yaml") == {}

    def test_parses_valid_yaml(self, tmp_path):
        f = tmp_path / "cfg.yaml"
        f.write_text("key: value\nnested:\n  a: 1\n")
        result = load_yaml_config(str(f))
        assert result["key"] == "value"
        assert result["nested"]["a"] == 1

    def test_empty_file_returns_empty_dict(self, tmp_path):
        f = tmp_path / "empty.yaml"
        f.write_text("")
        assert load_yaml_config(str(f)) == {}

    def test_multi_section_yaml(self, tmp_path):
        f = tmp_path / "multi.yaml"
        f.write_text(
            "llm:\n  use_local: true\n  temperature: 0.1\n"
            "chunking:\n  chunk_size: 512\n"
        )
        result = load_yaml_config(str(f))
        assert result["llm"]["use_local"] is True
        assert result["chunking"]["chunk_size"] == 512


class TestStructuredLogger:
    """
    StructuredLogger writes via Python's logging module.
    pytest intercepts log records — we use caplog (not capsys) to read them.
    caplog.records[n].getMessage() returns the raw JSON string we emit.
    """

    def _records(self, caplog, name):
        return [r for r in caplog.records if r.name == name]

    def test_info_emits_valid_json(self, caplog):
        name = "rag.test.sl.info"
        sl = StructuredLogger(name)
        with caplog.at_level(logging.INFO, logger=name):
            sl.info("hello", foo="bar")
        records = self._records(caplog, name)
        assert records, "no log records captured"
        data = json.loads(records[0].getMessage())
        assert data["message"] == "hello"
        assert data["level"] == "INFO"
        assert data["foo"] == "bar"
        assert "timestamp" in data

    def test_warning_level(self, caplog):
        name = "rag.test.sl.warning"
        sl = StructuredLogger(name)
        with caplog.at_level(logging.WARNING, logger=name):
            sl.warning("watch out", code=42)
        records = self._records(caplog, name)
        assert records
        data = json.loads(records[0].getMessage())
        assert data["level"] == "WARNING"
        assert data["code"] == 42

    def test_error_level(self, caplog):
        name = "rag.test.sl.error"
        sl = StructuredLogger(name)
        with caplog.at_level(logging.ERROR, logger=name):
            sl.error("something broke", detail="oops")
        records = self._records(caplog, name)
        assert records
        data = json.loads(records[0].getMessage())
        assert data["level"] == "ERROR"
        assert data["detail"] == "oops"

    def test_legacy_log_method(self, caplog):
        name = "rag.test.sl.legacy"
        sl = StructuredLogger(name)
        with caplog.at_level(logging.INFO, logger=name):
            sl.log("INFO", "legacy style", x=1)
        records = self._records(caplog, name)
        assert records
        data = json.loads(records[0].getMessage())
        assert data["message"] == "legacy style"
        assert data["x"] == 1

    def test_list_kwarg_serialised(self, caplog):
        name = "rag.test.sl.list"
        sl = StructuredLogger(name)
        with caplog.at_level(logging.INFO, logger=name):
            sl.info("msg", tags=["a", "b"])
        records = self._records(caplog, name)
        data = json.loads(records[0].getMessage())
        assert data["tags"] == ["a", "b"]

    def test_no_duplicate_handlers_on_reinit(self):
        name = "rag.test.sl.dedup"
        for _ in range(3):
            StructuredLogger(name)
        underlying = logging.getLogger(name)
        stream_handlers = [h for h in underlying.handlers
                           if isinstance(h, logging.StreamHandler)]
        assert len(stream_handlers) == 1
