"""
Experiment tracker for logging and comparing RAG configurations.

Each run is appended to results/experiments.jsonl (one JSON object per line)
and also saved as an individual results/run_<timestamp>.json file.
Optional MLflow backend is activated when config.yaml sets
experiment_tracking.backend = "mlflow".
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.utils import StructuredLogger, ensure_dir

logger = StructuredLogger(__name__)


class ExperimentTracker:
    """
    Lightweight JSON-based experiment tracker.

    Stores every run in two places so they are easy to query:
    - experiments.jsonl  — append-only log of all runs (fast pandas read)
    - run_<id>.json      — individual file per run (easy to read one at a time)
    """

    def __init__(
        self,
        results_dir: str = "results",
        use_mlflow: bool = False,
        mlflow_uri: str = "file:./mlruns",
    ) -> None:
        self.results_dir = ensure_dir(results_dir)
        self.runs_file = self.results_dir / "experiments.jsonl"
        self._mlflow: Any = None

        if use_mlflow:
            try:
                import mlflow  # type: ignore
                mlflow.set_tracking_uri(mlflow_uri)
                self._mlflow = mlflow
                logger.info("MLflow tracking enabled", uri=mlflow_uri)
            except ImportError:
                logger.warning("mlflow not installed — falling back to JSON tracking")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def log_run(
        self,
        config: Dict[str, Any],
        metrics: Dict[str, Any],
        tags: Optional[Dict[str, str]] = None,
    ) -> str:
        """
        Persist a single experiment run.

        Parameters
        ----------
        config:
            Hyperparameters for this run (chunk_size, top_k, strategy …).
        metrics:
            Evaluation results (hit_rate, mrr, ndcg, ragas scores …).
        tags:
            Free-form labels (e.g. {"note": "baseline"}).

        Returns
        -------
        str  run_id in YYYYMMDD_HHMMSS format.
        """
        run_id = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        run: Dict[str, Any] = {
            "run_id": run_id,
            "timestamp": datetime.utcnow().isoformat(),
            "config": config,
            "metrics": metrics,
            "tags": tags or {},
        }

        # Append to JSONL index
        with self.runs_file.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(run, ensure_ascii=False, default=str) + "\n")

        # Save individual file
        run_file = self.results_dir / f"run_{run_id}.json"
        with run_file.open("w", encoding="utf-8") as fh:
            json.dump(run, fh, indent=2, ensure_ascii=False, default=str)

        if self._mlflow is not None:
            self._log_to_mlflow(run)

        logger.info("Experiment run saved", run_id=run_id, file=str(run_file))
        return run_id

    def load_all_runs(self) -> List[Dict[str, Any]]:
        """Return all runs as a list, oldest first."""
        if not self.runs_file.exists():
            return []
        runs: List[Dict[str, Any]] = []
        with self.runs_file.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    try:
                        runs.append(json.loads(line))
                    except json.JSONDecodeError as exc:
                        logger.warning("Skipping malformed JSONL line", error=str(exc))
        return runs

    def get_best_run(self, metric: str, higher_is_better: bool = True) -> Optional[Dict[str, Any]]:
        """Return the run that maximises (or minimises) the given metric."""
        runs = [r for r in self.load_all_runs() if metric in r.get("metrics", {})]
        if not runs:
            return None
        key = lambda r: r["metrics"][metric]  # noqa: E731
        return max(runs, key=key) if higher_is_better else min(runs, key=key)

    def summary_table(self) -> str:
        """Return a markdown-formatted summary of all runs."""
        runs = self.load_all_runs()
        if not runs:
            return "No experiment runs recorded yet."

        header = "| run_id | chunk_size | strategy | hit_rate | mrr | ndcg | faithfulness |"
        sep = "|--------|-----------|----------|----------|-----|------|--------------|"
        rows = [header, sep]

        def _fmt(v: Any) -> str:
            return f"{v:.4f}" if isinstance(v, (int, float)) else "—"

        for r in runs:
            cfg = r.get("config", {})
            met = r.get("metrics", {})
            k = met.get("k", cfg.get("k", "?"))
            ndcg_key = f"ndcg_at_{k}"
            rows.append(
                f"| {r['run_id']} "
                f"| {cfg.get('chunk_size', '?')} "
                f"| {cfg.get('chunking_strategy', '?')} "
                f"| {_fmt(met.get('hit_rate_at_k'))} "
                f"| {_fmt(met.get('mrr'))} "
                f"| {_fmt(met.get(ndcg_key))} "
                f"| {_fmt(met.get('faithfulness'))} |"
            )

        return "\n".join(rows)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _log_to_mlflow(self, run: Dict[str, Any]) -> None:
        try:
            with self._mlflow.start_run(run_name=run["run_id"]):
                self._mlflow.log_params(
                    {k: v for k, v in run["config"].items() if isinstance(v, (str, int, float, bool))}
                )
                scalar_metrics = {
                    k: v for k, v in run["metrics"].items() if isinstance(v, (int, float))
                }
                if scalar_metrics:
                    self._mlflow.log_metrics(scalar_metrics)
                for k, v in run.get("tags", {}).items():
                    self._mlflow.set_tag(k, v)
        except Exception as exc:
            logger.warning("MLflow logging failed", error=str(exc))
