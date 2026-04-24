"""
Retrieval quality benchmark: Hit Rate@k, MRR, NDCG@k.

Methodology
-----------
Each TestCase carries a question, expected ground-truth answer, and a set of
``relevant_keywords``. A retrieved chunk is considered *relevant* when it
contains at least one keyword (case-insensitive). This proxy avoids the need
for manually labelled chunk IDs while still producing meaningful signals for
comparing retrieval configurations.

Metrics
-------
Hit Rate@k  — fraction of questions where ≥1 relevant chunk appears in top-k
              (recall proxy; tells you whether the system *can* answer the question)
MRR         — mean reciprocal rank of the first relevant result
              (measures how quickly the system surfaces the answer)
NDCG@k      — normalised discounted cumulative gain
              (full ranking-quality metric that penalises relevant results ranked low)
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from src.utils import StructuredLogger

logger = StructuredLogger(__name__)


@dataclass
class TestCase:
    """Single evaluation question with expected answer metadata."""

    id: str
    question: str
    ground_truth: str
    relevant_keywords: List[str]
    category: str = "general"
    source_document: Optional[str] = None


@dataclass
class RetrievalResult:
    """Per-question retrieval evaluation result."""

    test_case_id: str
    question: str
    hit_at_k: bool
    reciprocal_rank: float
    ndcg_at_k: float
    num_retrieved: int


def load_test_cases(path: str) -> List[TestCase]:
    """Load test cases from a JSON file."""
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    return [TestCase(**item) for item in raw]


def _chunk_is_relevant(chunk_text: str, keywords: List[str]) -> bool:
    """Return True when chunk contains at least one keyword (case-insensitive)."""
    text_lower = chunk_text.lower()
    return any(kw.lower() in text_lower for kw in keywords)


def _dcg(relevances: List[int]) -> float:
    """Discounted Cumulative Gain."""
    return sum(rel / math.log2(idx + 2) for idx, rel in enumerate(relevances))


def compute_ndcg(relevances: List[int], k: int) -> float:
    """NDCG@k. Ideal ranking is the sorted (descending) relevance list."""
    ideal = sorted(relevances, reverse=True)[:k]
    actual = relevances[:k]
    ideal_dcg = _dcg(ideal)
    if ideal_dcg == 0:
        return 0.0
    return _dcg(actual) / ideal_dcg


def evaluate_retrieval(
    test_cases: List[TestCase],
    retriever_fn: Callable[[str], List[str]],
    k: int = 5,
) -> Dict[str, Any]:
    """
    Run the retrieval benchmark over all test cases.

    Parameters
    ----------
    test_cases:
        Questions with ground-truth keyword signals.
    retriever_fn:
        Callable that accepts a query string and returns a list of chunk
        texts ordered by retrieval score (best first, length ≤ k).
    k:
        Evaluation cutoff.

    Returns
    -------
    Dict with aggregate metrics (hit_rate_at_k, mrr, ndcg_at_k) and
    per-question breakdowns.
    """
    results: List[RetrievalResult] = []

    for tc in test_cases:
        logger.info("Evaluating", question_id=tc.id)
        try:
            chunks = retriever_fn(tc.question)[:k]
        except Exception as exc:
            logger.error("Retriever failed", question_id=tc.id, error=str(exc))
            chunks = []

        relevances = [
            1 if _chunk_is_relevant(c, tc.relevant_keywords) else 0
            for c in chunks
        ]

        hit = any(r == 1 for r in relevances)

        rr = 0.0
        for rank, rel in enumerate(relevances, start=1):
            if rel == 1:
                rr = 1.0 / rank
                break

        ndcg = compute_ndcg(relevances, k)

        results.append(
            RetrievalResult(
                test_case_id=tc.id,
                question=tc.question,
                hit_at_k=hit,
                reciprocal_rank=rr,
                ndcg_at_k=ndcg,
                num_retrieved=len(chunks),
            )
        )

    n = len(results)
    if n == 0:
        return {"error": "No test cases evaluated", "results": []}

    hit_rate = sum(r.hit_at_k for r in results) / n
    mrr = sum(r.reciprocal_rank for r in results) / n
    ndcg_mean = sum(r.ndcg_at_k for r in results) / n

    logger.info(
        "Retrieval benchmark complete",
        n=n,
        hit_rate=round(hit_rate, 4),
        mrr=round(mrr, 4),
        ndcg=round(ndcg_mean, 4),
    )

    return {
        "hit_rate_at_k": round(hit_rate, 4),
        "mrr": round(mrr, 4),
        f"ndcg_at_{k}": round(ndcg_mean, 4),
        "k": k,
        "n_questions": n,
        "per_question": [
            {
                "id": r.test_case_id,
                "question": r.question,
                "hit": r.hit_at_k,
                "rr": r.reciprocal_rank,
                "ndcg": r.ndcg_at_k,
            }
            for r in results
        ],
    }
