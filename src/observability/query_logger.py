"""
SQLite-based query logger for production observability.

Persists every query with its retrieved chunks, generated answer, and latency.
Use metrics.py or any SQLite browser to inspect the log.

WHY SQLite
----------
Zero extra dependencies, embeds directly in the project, easy to query with
pandas or a GUI. Schema is simple enough to migrate to Postgres or BigQuery
without application changes.
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Generator, List

from src.utils import StructuredLogger, ensure_dir

logger = StructuredLogger(__name__)

_DDL = """
CREATE TABLE IF NOT EXISTS queries (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp       TEXT    NOT NULL,
    question        TEXT    NOT NULL,
    answer          TEXT    NOT NULL,
    sources         TEXT    NOT NULL,     -- JSON array of source dicts
    latency_ms      REAL    NOT NULL,
    num_sources     INTEGER NOT NULL,
    retrieval_mode  TEXT    NOT NULL,
    document_filter TEXT                  -- JSON array or NULL
)
"""


class QueryLogger:
    """Thread-safe SQLite query logger."""

    def __init__(self, db_path: str = "logs/queries.db") -> None:
        self.db_path = Path(db_path)
        ensure_dir(self.db_path.parent)
        self._init_db()
        logger.info("QueryLogger initialised", db=str(self.db_path))

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(_DDL)

    @contextmanager
    def _connect(self) -> Generator[sqlite3.Connection, None, None]:
        conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def log_query(self, question: str, result: Dict[str, Any]) -> None:
        """Persist a query result to SQLite."""
        meta = result.get("metadata", {})
        try:
            with self._connect() as conn:
                conn.execute(
                    """
                    INSERT INTO queries
                        (timestamp, question, answer, sources, latency_ms,
                         num_sources, retrieval_mode, document_filter)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        datetime.utcnow().isoformat(),
                        question,
                        result.get("answer", ""),
                        json.dumps(result.get("sources", []), default=str),
                        round(meta.get("duration_seconds", 0) * 1000, 2),
                        meta.get("num_sources", 0),
                        meta.get("retrieval_mode", "unknown"),
                        json.dumps(meta.get("document_filter"))
                        if meta.get("document_filter")
                        else None,
                    ),
                )
        except Exception as exc:
            logger.error("Failed to log query", error=str(exc))

    def get_summary(self) -> Dict[str, Any]:
        """Return aggregate statistics from the query log."""
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT
                    COUNT(*)         AS total_queries,
                    AVG(latency_ms)  AS avg_latency_ms,
                    MIN(latency_ms)  AS min_latency_ms,
                    MAX(latency_ms)  AS max_latency_ms,
                    AVG(num_sources) AS avg_sources
                FROM queries
                """
            ).fetchone()

            top_qs = conn.execute(
                """
                SELECT question, COUNT(*) AS cnt
                FROM queries
                GROUP BY question
                ORDER BY cnt DESC
                LIMIT 10
                """
            ).fetchall()

        return {
            "total_queries": row["total_queries"] or 0,
            "avg_latency_ms": round(row["avg_latency_ms"] or 0, 1),
            "min_latency_ms": round(row["min_latency_ms"] or 0, 1),
            "max_latency_ms": round(row["max_latency_ms"] or 0, 1),
            "avg_sources_retrieved": round(row["avg_sources"] or 0, 1),
            "top_10_questions": [
                {"question": q["question"], "count": q["cnt"]} for q in top_qs
            ],
        }

    def get_recent_queries(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Fetch the most recent queries (newest first)."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM queries ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()
        return [dict(r) for r in rows]
