"""
Shared utilities: structured logging and filesystem helpers.

WHY: StructuredLogger was duplicated in engine.py and ingestor.py.
     Extracting to a shared module eliminates the duplication and
     ensures all components emit identical log schemas.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Union


class StructuredLogger:
    """
    JSON-formatted logger for production traceability.

    Output format is directly parseable by ELK, Datadog, and Grafana.
    Supports both the legacy .log(level, msg, **kw) call style and
    idiomatic .info() / .warning() / .error() / .debug() shortcuts.
    """

    def __init__(self, name: str, level: str = "INFO") -> None:
        self.logger = logging.getLogger(name)
        numeric_level = getattr(logging, level.upper(), logging.INFO)
        self.logger.setLevel(numeric_level)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter("%(message)s"))
            self.logger.addHandler(handler)

    def _emit(self, level: str, message: str, **kwargs: Any) -> None:
        entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": level,
            "message": message,
            **kwargs,
        }
        log_fn = getattr(self.logger, level.lower(), self.logger.info)
        log_fn(json.dumps(entry, ensure_ascii=False, default=str))

    # Idiomatic shortcuts
    def info(self, message: str, **kwargs: Any) -> None:
        self._emit("INFO", message, **kwargs)

    def warning(self, message: str, **kwargs: Any) -> None:
        self._emit("WARNING", message, **kwargs)

    def error(self, message: str, **kwargs: Any) -> None:
        self._emit("ERROR", message, **kwargs)

    def debug(self, message: str, **kwargs: Any) -> None:
        self._emit("DEBUG", message, **kwargs)

    # Backward-compatible call style used throughout engine.py / ingestor.py
    def log(self, level: str, message: str, **kwargs: Any) -> None:
        self._emit(level, message, **kwargs)


def ensure_dir(path: Union[str, Path]) -> Path:
    """Create directory (and all parents) if it does not exist."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def load_yaml_config(path: Union[str, Path]) -> dict:
    """
    Load a YAML config file. Returns an empty dict if the file is missing.

    WHY: Graceful degradation lets RAGConfig fall back to its hardcoded
         defaults when config.yaml has not yet been created.
    """
    try:
        import yaml  # PyYAML — optional at import time
        with open(path, encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except FileNotFoundError:
        return {}
    except Exception as exc:
        logging.getLogger(__name__).warning(
            json.dumps({"message": f"Could not load {path}: {exc}"})
        )
        return {}
