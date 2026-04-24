"""Tests for src/evaluation/benchmark.py — no ML dependencies required."""
import math

import pytest

from src.evaluation.benchmark import (
    TestCase,
    _chunk_is_relevant,
    _dcg,
    compute_ndcg,
    evaluate_retrieval,
)


def _case(idx: int = 1, keywords=("python",)) -> TestCase:
    return TestCase(
        id=str(idx),
        question=f"Question {idx}",
        ground_truth="Answer",
        relevant_keywords=list(keywords),
    )


# ── _chunk_is_relevant ──────────────────────────────────────────────────────────

class TestChunkIsRelevant:
    def test_exact_match(self):
        assert _chunk_is_relevant("attention mechanism in transformers", ["attention"])

    def test_case_insensitive(self):
        assert _chunk_is_relevant("The ATTENTION MECHANISM", ["attention"])

    def test_no_match(self):
        assert not _chunk_is_relevant("transformer model architecture", ["attention"])

    def test_any_keyword_matches(self):
        assert _chunk_is_relevant("BERT language model", ["attention", "BERT"])

    def test_empty_keywords_returns_false(self):
        assert not _chunk_is_relevant("some text", [])

    def test_substring_match(self):
        assert _chunk_is_relevant("self-attention mechanism", ["attention"])


# ── _dcg ───────────────────────────────────────────────────────────────────────

class TestDCG:
    def test_single_relevant(self):
        assert _dcg([1]) == pytest.approx(1.0)

    def test_all_zeros(self):
        assert _dcg([0, 0, 0]) == 0.0

    def test_first_position_beats_second(self):
        assert _dcg([1, 0]) > _dcg([0, 1])

    def test_perfect_three_hits(self):
        expected = 1.0 + 1 / math.log2(3) + 1 / math.log2(4)
        assert _dcg([1, 1, 1]) == pytest.approx(expected)


# ── compute_ndcg ───────────────────────────────────────────────────────────────

class TestComputeNDCG:
    def test_perfect_score(self):
        assert compute_ndcg([1, 1, 0], k=3) == pytest.approx(1.0)

    def test_zero_score(self):
        assert compute_ndcg([0, 0, 0], k=3) == 0.0

    def test_suboptimal_ordering_between_zero_and_one(self):
        score = compute_ndcg([0, 1, 0], k=3)
        assert 0 < score < 1.0

    def test_k_truncates_beyond_cutoff(self):
        # Relevant chunk only at position 3 (index 2) is outside k=2
        assert compute_ndcg([0, 0, 1], k=2) == 0.0

    def test_k_equals_one(self):
        assert compute_ndcg([1], k=1) == pytest.approx(1.0)
        assert compute_ndcg([0], k=1) == 0.0


# ── evaluate_retrieval ─────────────────────────────────────────────────────────

class TestEvaluateRetrieval:
    def test_all_hits(self):
        result = evaluate_retrieval(
            [_case(keywords=["python"])],
            lambda q: ["python is great"],
            k=2,
        )
        assert result["hit_rate_at_k"] == 1.0
        assert result["mrr"] == 1.0

    def test_no_hits(self):
        result = evaluate_retrieval(
            [_case(keywords=["python"])],
            lambda q: ["java and ruby"],
            k=2,
        )
        assert result["hit_rate_at_k"] == 0.0
        assert result["mrr"] == 0.0

    def test_partial_hit_rate(self):
        cases = [_case(1, ["python"]), _case(2, ["java"])]
        result = evaluate_retrieval(cases, lambda q: ["python chunk"], k=5)
        assert result["hit_rate_at_k"] == pytest.approx(0.5)

    def test_mrr_at_second_position(self):
        result = evaluate_retrieval(
            [_case(keywords=["python"])],
            lambda q: ["irrelevant", "python rocks"],
            k=5,
        )
        assert result["mrr"] == pytest.approx(0.5)

    def test_empty_cases_returns_error(self):
        result = evaluate_retrieval([], lambda q: [], k=5)
        assert "error" in result

    def test_n_questions_reported(self):
        cases = [_case(i) for i in range(3)]
        result = evaluate_retrieval(cases, lambda q: [], k=5)
        assert result["n_questions"] == 3

    def test_k_reported_in_result(self):
        result = evaluate_retrieval([_case()], lambda q: [], k=7)
        assert result["k"] == 7

    def test_retriever_exception_counts_as_miss(self):
        def bad_retriever(q):
            raise RuntimeError("retriever down")

        result = evaluate_retrieval([_case(keywords=["python"])], bad_retriever, k=5)
        assert result["hit_rate_at_k"] == 0.0

    def test_per_question_breakdown_included(self):
        result = evaluate_retrieval(
            [_case(1, ["python"]), _case(2, ["java"])],
            lambda q: ["python chunk"],
            k=3,
        )
        assert "per_question" in result
        assert len(result["per_question"]) == 2

    def test_ndcg_key_uses_k(self):
        result = evaluate_retrieval([_case()], lambda q: [], k=5)
        assert "ndcg_at_5" in result
