"""
RAG Evaluation Suite.

Runs two complementary evaluation passes:

1. Retrieval Quality  — Hit Rate@k, MRR, NDCG@k (no LLM required).
   Uses the configured retrieval strategy (hybrid or vector-only) to
   rank chunks for each test question and measures whether the gold
   keywords appear in the top-k results.

2. Answer Quality (RAGAS-style) — Answer Relevancy, Faithfulness,
   Context Precision, Context Recall.
   Requires a live LLM (LM Studio by default) to generate answers.
   Skip with --retrieval-only when the LLM is unavailable.

Results are saved to results/run_<timestamp>.json and appended to
results/experiments.jsonl for later comparison in the notebook.

Usage
-----
  # Full eval (needs LM Studio + ingested index)
  python evaluate.py

  # Retrieval-only (no LLM needed)
  python evaluate.py --retrieval-only

  # Override config and k
  python evaluate.py --config config.yaml --k 5

  # Tag the run for later filtering
  python evaluate.py --tag "baseline_sentence_512"
  python evaluate.py --tag "fixed_256" --config experiments/fixed_256.yaml
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

# Allow running from project root without installing the package
sys.path.insert(0, str(Path(__file__).parent))

from src.config import RAGConfig
from src.evaluation.benchmark import TestCase, load_test_cases, evaluate_retrieval
from src.evaluation.ragas_eval import evaluate_dataset
from src.experiment_tracker import ExperimentTracker
from src.utils import StructuredLogger, ensure_dir

logger = StructuredLogger(__name__)


# ---------------------------------------------------------------------------
# Retriever helper
# ---------------------------------------------------------------------------

def build_retriever_fn(engine: Any, k: int):
    """
    Wrap the engine's vector index in a simple callable for the benchmark.

    Uses the raw VectorIndexRetriever (not the full hybrid pipeline) so that
    the benchmark measures pure retrieval quality, independent of the fusion
    or reranking post-processors.
    """
    from llama_index.core.retrievers import VectorIndexRetriever

    def retriever_fn(question: str) -> List[str]:
        try:
            retriever = VectorIndexRetriever(
                index=engine.index,
                similarity_top_k=k,
            )
            nodes = retriever.retrieve(question)
            return [n.node.text for n in nodes]
        except Exception as exc:
            logger.error("Retrieval failed", error=str(exc))
            return []

    return retriever_fn


# ---------------------------------------------------------------------------
# RAGAS eval helper
# ---------------------------------------------------------------------------

def run_ragas_eval(
    engine: Any,
    test_cases: List[TestCase],
) -> Dict[str, Any]:
    """
    Query the engine for each test case and compute answer-quality metrics.

    Falls back gracefully when the LLM call fails (e.g. LM Studio not loaded).
    """
    samples: List[Dict[str, Any]] = []

    for tc in test_cases:
        try:
            result = engine.query(tc.question)
            if "error" in result.get("metadata", {}):
                logger.warning("LLM returned error for question", question_id=tc.id)
                continue
            samples.append(
                {
                    "question": tc.question,
                    "answer": result["answer"],
                    "contexts": [s["text_preview"] for s in result["sources"]],
                    "ground_truth": tc.ground_truth,
                }
            )
        except Exception as exc:
            logger.error("RAGAS sample failed", question_id=tc.id, error=str(exc))

    if not samples:
        return {"error": "No samples could be evaluated — is the LLM running?"}

    return evaluate_dataset(samples, embed_fn=engine.embed_model.get_text_embedding)


# ---------------------------------------------------------------------------
# Report printer
# ---------------------------------------------------------------------------

def print_report(
    retrieval_metrics: Dict[str, Any],
    ragas_metrics: Dict[str, Any],
    config: RAGConfig,
    run_id: str,
) -> None:
    k = retrieval_metrics.get("k", config.vector_top_k)
    ndcg_key = f"ndcg_at_{k}"

    print()
    print("=" * 62)
    print("  RAG EVALUATION REPORT")
    print("=" * 62)
    print(f"  Run ID : {run_id}")
    print(f"  Time   : {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC")
    print()
    print("  Configuration")
    print(f"    Chunking   : {config.chunking_strategy}  "
          f"(size={config.chunk_size}, overlap={config.chunk_overlap})")
    print(f"    Retrieval  : vector_top_k={config.vector_top_k}  "
          f"bm25_top_k={config.bm25_top_k}")
    print(f"    Reranking  : {'enabled' if config.use_cross_encoder else 'disabled'}")
    print()
    print(f"  Retrieval Quality  (k={k}, n={retrieval_metrics.get('n_questions', '?')})")
    print(f"    Hit Rate@{k}  : {retrieval_metrics.get('hit_rate_at_k', 0.0):.4f}")
    print(f"    MRR         : {retrieval_metrics.get('mrr', 0.0):.4f}")
    print(f"    NDCG@{k}     : {retrieval_metrics.get(ndcg_key, 0.0):.4f}")

    if ragas_metrics and "error" not in ragas_metrics:
        n = ragas_metrics.get("n_samples", "?")
        print()
        print(f"  Answer Quality (RAGAS-style, n={n})")
        print(f"    Answer Relevancy  : {ragas_metrics.get('answer_relevancy', 0.0):.4f}")
        print(f"    Faithfulness      : {ragas_metrics.get('faithfulness', 0.0):.4f}")
        print(f"    Context Precision : {ragas_metrics.get('context_precision', 0.0):.4f}")
        print(f"    Context Recall    : {ragas_metrics.get('context_recall', 0.0):.4f}")
    elif ragas_metrics and "error" in ragas_metrics:
        print()
        print(f"  Answer Quality : SKIPPED — {ragas_metrics['error']}")

    print()
    print("=" * 62)


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the full RAG evaluation suite.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    parser.add_argument(
        "--test-questions",
        default="data/test_questions.json",
        help="Path to test questions JSON",
    )
    parser.add_argument("--k", type=int, default=5, help="Retrieval cutoff for evaluation")
    parser.add_argument(
        "--retrieval-only",
        action="store_true",
        help="Skip RAGAS answer-quality eval (no LLM required)",
    )
    parser.add_argument("--tag", type=str, default="", help="Free-text tag for this run")
    args = parser.parse_args()

    # ---- Config ----
    config = RAGConfig.from_yaml(args.config)
    ensure_dir(config.results_dir)

    # ---- Test data ----
    test_path = Path(args.test_questions)
    if not test_path.exists():
        print(f"ERROR: test questions not found at {test_path}")
        print("       Expected format: data/test_questions.json")
        sys.exit(1)

    test_cases = load_test_cases(str(test_path))
    print(f"Loaded {len(test_cases)} test cases from {test_path}")

    # ---- Engine init ----
    print("Initialising RAG engine …")
    try:
        from src.engine import RAGQueryEngine
        engine = RAGQueryEngine(config=config, use_hybrid=True, use_reranking=False)
    except Exception as exc:
        print(f"\nERROR: Could not initialise the RAG engine: {exc}")
        print(
            "       Ensure the vector index exists (run: python src/ingestor.py --recreate)\n"
            "       and, for RAGAS eval, that LM Studio is running at "
            f"{config.lm_studio_base_url}"
        )
        sys.exit(1)

    # ---- Retrieval benchmark ----
    print(f"\nRunning retrieval benchmark (k={args.k}) …")
    retriever_fn = build_retriever_fn(engine, k=args.k)
    retrieval_metrics = evaluate_retrieval(test_cases, retriever_fn, k=args.k)

    # ---- RAGAS eval ----
    ragas_metrics: Dict[str, Any] = {}
    if not args.retrieval_only:
        print("Running RAGAS answer-quality evaluation …")
        try:
            ragas_metrics = run_ragas_eval(engine, test_cases)
        except Exception as exc:
            ragas_metrics = {"error": str(exc)}
            logger.warning("RAGAS eval raised exception", error=str(exc))
    else:
        print("Skipping RAGAS eval (--retrieval-only flag set).")

    # ---- Experiment tracking ----
    tracker = ExperimentTracker(results_dir=config.results_dir)
    exp_config = {
        "chunking_strategy": config.chunking_strategy,
        "chunk_size": config.chunk_size,
        "chunk_overlap": config.chunk_overlap,
        "vector_top_k": config.vector_top_k,
        "bm25_top_k": config.bm25_top_k,
        "rerank_top_n": config.rerank_top_n,
        "use_cross_encoder": config.use_cross_encoder,
        "k": args.k,
    }
    all_metrics: Dict[str, Any] = {**retrieval_metrics}
    if ragas_metrics and "error" not in ragas_metrics:
        # Flatten aggregate scores only (exclude per_sample list)
        for key in ("answer_relevancy", "faithfulness", "context_precision", "context_recall", "n_samples"):
            if key in ragas_metrics:
                all_metrics[key] = ragas_metrics[key]

    tags = {}
    if args.tag:
        tags["tag"] = args.tag

    run_id = tracker.log_run(config=exp_config, metrics=all_metrics, tags=tags)

    # ---- Save full report ----
    report = {
        "run_id": run_id,
        "timestamp": datetime.utcnow().isoformat(),
        "tag": args.tag,
        "config": exp_config,
        "retrieval_metrics": retrieval_metrics,
        "ragas_metrics": ragas_metrics,
    }
    report_path = Path(config.results_dir) / f"eval_{run_id}.json"
    with report_path.open("w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False, default=str)

    # ---- Print ----
    print_report(retrieval_metrics, ragas_metrics, config, run_id)
    print(f"  Results : {report_path}")
    print(f"  Index   : {Path(config.results_dir) / 'experiments.jsonl'}")
    print()


if __name__ == "__main__":
    main()
