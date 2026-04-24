"""
RAGAS-inspired answer-quality metrics.

Metrics
-------
answer_relevancy   cosine_similarity(embed(question), embed(answer))
                   Measures whether the answer actually addresses the question.
                   Range [0, 1]; typical good answers score 0.6–0.95.

faithfulness       fraction of answer sentences whose keywords are grounded
                   in the retrieved context.
                   Range [0, 1]; 1.0 = fully grounded, 0.0 = hallucinated.

context_precision  fraction of retrieved chunks that are relevant to the question.
                   Range [0, 1]; measures retrieval precision.

context_recall     keyword overlap between the combined retrieved context and
                   the ground-truth answer.
                   Range [0, 1]; measures retrieval coverage.

WHY a custom implementation instead of the ragas library
---------------------------------------------------------
The official ragas package pulls in langchain + openai as hard dependencies,
which conflict with llama-index 0.10 in some environments. These manual
implementations are transparent, dependency-free, and use the same embedding
model as the main RAG pipeline — ensuring metric consistency.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, List

from src.utils import StructuredLogger

logger = StructuredLogger(__name__)

# Words that carry no signal for keyword-overlap calculations
_STOP_WORDS = {
    "the", "a", "an", "is", "are", "was", "were", "to", "of", "and", "or",
    "in", "on", "at", "for", "with", "by", "from", "that", "this", "it",
    "be", "as", "not", "its", "their", "they", "we", "you", "he", "she",
    "has", "have", "had", "do", "does", "did", "will", "would", "can",
    "could", "should", "which", "who", "what", "when", "where", "how",
}


# ============================================================================
# LOW-LEVEL HELPERS
# ============================================================================

def _cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a ** 2 for a in vec_a))
    norm_b = math.sqrt(sum(b ** 2 for b in vec_b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def _meaningful_words(text: str) -> set:
    """Lowercase alphanumeric tokens, stop-words removed."""
    return {
        w.lower()
        for w in re.findall(r"\w+", text)
        if w.lower() not in _STOP_WORDS and len(w) > 2
    }


def _keyword_overlap(text: str, reference: str) -> float:
    """Fraction of meaningful reference words found in text."""
    ref_words = _meaningful_words(reference)
    if not ref_words:
        return 0.0
    text_words = _meaningful_words(text)
    return len(ref_words & text_words) / len(ref_words)


def _split_sentences(text: str) -> List[str]:
    """Split text on sentence-ending punctuation."""
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s.strip() for s in sentences if s.strip()]


# ============================================================================
# PER-METRIC FUNCTIONS (public API)
# ============================================================================

def compute_answer_relevancy(
    question: str,
    answer: str,
    embed_fn: Callable[[str], List[float]],
) -> float:
    """
    Semantic similarity between question and answer embeddings.

    A high score means the answer is topically aligned with the question.
    A low score suggests the answer drifted off-topic or is empty.
    """
    try:
        q_emb = embed_fn(question)
        a_emb = embed_fn(answer)
        return max(0.0, _cosine_similarity(q_emb, a_emb))
    except Exception as exc:
        logger.error("answer_relevancy failed", error=str(exc))
        return 0.0


def compute_faithfulness(answer: str, contexts: List[str]) -> float:
    """
    Fraction of answer sentences that are grounded in retrieved context.

    A sentence is considered grounded when its keyword overlap with the
    combined context exceeds the 0.4 threshold.
    """
    sentences = _split_sentences(answer)
    if not sentences:
        return 0.0
    combined = " ".join(contexts)
    supported = sum(
        1 for s in sentences if _keyword_overlap(combined, s) >= 0.4
    )
    return supported / len(sentences)


def compute_context_precision(question: str, contexts: List[str]) -> float:
    """
    Fraction of retrieved chunks that are relevant to the question.

    A chunk is relevant when its keyword overlap with the question ≥ 0.2.
    """
    if not contexts:
        return 0.0
    relevant = sum(1 for ctx in contexts if _keyword_overlap(ctx, question) >= 0.2)
    return relevant / len(contexts)


def compute_context_recall(ground_truth: str, contexts: List[str]) -> float:
    """
    Fraction of ground-truth information covered by the retrieved context.

    Approximated as keyword overlap between combined context and ground truth.
    """
    if not contexts:
        return 0.0
    return _keyword_overlap(" ".join(contexts), ground_truth)


# ============================================================================
# COMPOSITE EVALUATION
# ============================================================================

@dataclass
class RAGASResult:
    """RAGAS-style quality scores for a single QA example."""

    question: str
    answer: str
    answer_relevancy: float
    faithfulness: float
    context_precision: float
    context_recall: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "question": self.question,
            "answer_snippet": self.answer[:120],
            "answer_relevancy": round(self.answer_relevancy, 4),
            "faithfulness": round(self.faithfulness, 4),
            "context_precision": round(self.context_precision, 4),
            "context_recall": round(self.context_recall, 4),
        }


def evaluate_sample(
    question: str,
    answer: str,
    contexts: List[str],
    ground_truth: str,
    embed_fn: Callable[[str], List[float]],
) -> RAGASResult:
    """Compute all four metrics for a single QA sample."""
    return RAGASResult(
        question=question,
        answer=answer,
        answer_relevancy=compute_answer_relevancy(question, answer, embed_fn),
        faithfulness=compute_faithfulness(answer, contexts),
        context_precision=compute_context_precision(question, contexts),
        context_recall=compute_context_recall(ground_truth, contexts),
    )


def evaluate_dataset(
    samples: List[Dict[str, Any]],
    embed_fn: Callable[[str], List[float]],
) -> Dict[str, Any]:
    """
    Run RAGAS-style evaluation over a list of QA samples.

    Each sample dict must contain:
        question    (str)
        answer      (str)
        contexts    (List[str])
        ground_truth (str)

    Returns aggregate metrics and per-sample breakdowns.
    """
    results: List[RAGASResult] = []

    for i, s in enumerate(samples):
        logger.info("RAGAS eval", sample=i, question=s["question"][:60])
        try:
            result = evaluate_sample(
                question=s["question"],
                answer=s["answer"],
                contexts=s.get("contexts", []),
                ground_truth=s.get("ground_truth", ""),
                embed_fn=embed_fn,
            )
            results.append(result)
        except Exception as exc:
            logger.error("Sample eval failed", sample=i, error=str(exc))

    n = len(results)
    if n == 0:
        return {"error": "No samples evaluated"}

    aggregate = {
        "answer_relevancy": round(sum(r.answer_relevancy for r in results) / n, 4),
        "faithfulness": round(sum(r.faithfulness for r in results) / n, 4),
        "context_precision": round(sum(r.context_precision for r in results) / n, 4),
        "context_recall": round(sum(r.context_recall for r in results) / n, 4),
        "n_samples": n,
        "per_sample": [r.to_dict() for r in results],
    }

    logger.info(
        "RAGAS evaluation complete",
        **{k: v for k, v in aggregate.items() if k != "per_sample"},
    )
    return aggregate
