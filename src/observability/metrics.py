"""
CLI metrics viewer for the RAG system query log.

Usage
-----
    python -m src.observability.metrics
    python -m src.observability.metrics --db logs/queries.db --recent 20
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running as `python src/observability/metrics.py` from project root
_root = Path(__file__).resolve().parent.parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from src.observability.query_logger import QueryLogger


def main() -> None:
    parser = argparse.ArgumentParser(description="RAG observability metrics")
    parser.add_argument(
        "--db", default="logs/queries.db", help="Path to SQLite query log"
    )
    parser.add_argument(
        "--recent", type=int, default=10, help="Number of recent queries to display"
    )
    args = parser.parse_args()

    if not Path(args.db).exists():
        print(f"Query log not found: {args.db}")
        print("Run the app and submit some queries first.")
        return

    ql = QueryLogger(db_path=args.db)
    summary = ql.get_summary()
    recent = ql.get_recent_queries(limit=args.recent)

    print("\n" + "=" * 56)
    print("  RAG System — Observability Metrics")
    print("=" * 56)
    print(f"  Total queries logged  : {summary['total_queries']}")
    print(f"  Avg latency           : {summary['avg_latency_ms']} ms")
    print(f"  Min / Max latency     : {summary['min_latency_ms']} / {summary['max_latency_ms']} ms")
    print(f"  Avg sources returned  : {summary['avg_sources_retrieved']}")

    print("\n  --- Top Questions ---")
    for item in summary["top_10_questions"]:
        q = item["question"][:72]
        print(f"  [{item['count']:3d}x] {q}")

    print(f"\n  --- {args.recent} Most Recent Queries ---")
    for row in recent:
        ts = row["timestamp"][:19]
        q = row["question"][:58]
        lat = row["latency_ms"]
        print(f"  {ts}  {lat:8.1f} ms  {q}")

    print()


if __name__ == "__main__":
    main()
