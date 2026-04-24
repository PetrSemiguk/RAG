"""Tests for src/observability/query_logger.py — no ML dependencies required."""
import pytest

from src.observability.query_logger import QueryLogger


@pytest.fixture
def logger(tmp_path):
    return QueryLogger(db_path=str(tmp_path / "test_queries.db"))


def _result(answer="ok", duration=0.5, mode="hybrid", n_sources=2, sources=None):
    src = sources or [{"file_name": f"doc{i}.pdf"} for i in range(n_sources)]
    return {
        "answer": answer,
        "sources": src,
        "metadata": {
            "duration_seconds": duration,
            "num_sources": len(src),
            "retrieval_mode": mode,
            "document_filter": None,
        },
    }


class TestInit:
    def test_creates_db_file(self, tmp_path):
        db = tmp_path / "queries.db"
        QueryLogger(db_path=str(db))
        assert db.exists()

    def test_creates_parent_dirs(self, tmp_path):
        db = tmp_path / "nested" / "dir" / "queries.db"
        QueryLogger(db_path=str(db))
        assert db.exists()


class TestLogAndRetrieve:
    def test_log_persists_question_and_answer(self, logger):
        logger.log_query("What is RAG?", _result("RAG is…"))
        recent = logger.get_recent_queries(limit=1)
        assert recent[0]["question"] == "What is RAG?"
        assert recent[0]["answer"] == "RAG is…"

    def test_retrieval_mode_stored(self, logger):
        logger.log_query("Q", _result(mode="vector"))
        assert logger.get_recent_queries(limit=1)[0]["retrieval_mode"] == "vector"

    def test_latency_converted_to_ms(self, logger):
        logger.log_query("Q", _result(duration=1.5))
        assert logger.get_recent_queries(limit=1)[0]["latency_ms"] == pytest.approx(1500.0, abs=1.0)

    def test_num_sources_stored(self, logger):
        logger.log_query("Q", _result(n_sources=3))
        assert logger.get_recent_queries(limit=1)[0]["num_sources"] == 3

    def test_recent_queries_newest_first(self, logger):
        logger.log_query("first", _result())
        logger.log_query("second", _result())
        recent = logger.get_recent_queries(limit=2)
        assert recent[0]["question"] == "second"

    def test_limit_respected(self, logger):
        for i in range(5):
            logger.log_query(f"Q{i}", _result())
        assert len(logger.get_recent_queries(limit=3)) == 3


class TestSummary:
    def test_empty_db(self, logger):
        s = logger.get_summary()
        assert s["total_queries"] == 0
        assert s["avg_latency_ms"] == 0.0

    def test_total_queries_count(self, logger):
        logger.log_query("Q1", _result(duration=0.1))
        logger.log_query("Q2", _result(duration=0.3))
        assert logger.get_summary()["total_queries"] == 2

    def test_avg_latency(self, logger):
        logger.log_query("Q1", _result(duration=0.1))
        logger.log_query("Q2", _result(duration=0.3))
        # 100 ms + 300 ms → avg 200 ms
        assert logger.get_summary()["avg_latency_ms"] == pytest.approx(200.0, abs=1.0)

    def test_min_max_latency(self, logger):
        logger.log_query("Q1", _result(duration=0.1))
        logger.log_query("Q2", _result(duration=0.9))
        s = logger.get_summary()
        assert s["min_latency_ms"] == pytest.approx(100.0, abs=1.0)
        assert s["max_latency_ms"] == pytest.approx(900.0, abs=1.0)

    def test_top_questions_ordered_by_count(self, logger):
        for _ in range(3):
            logger.log_query("repeat", _result())
        logger.log_query("unique", _result())
        top = logger.get_summary()["top_10_questions"]
        assert top[0]["question"] == "repeat"
        assert top[0]["count"] == 3
