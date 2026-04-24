"""Tests for src/experiment_tracker.py — no ML dependencies required."""
import pytest

from src.experiment_tracker import ExperimentTracker


@pytest.fixture
def tracker(tmp_path):
    return ExperimentTracker(results_dir=str(tmp_path))


class TestLogRun:
    def test_creates_jsonl_index(self, tracker, tmp_path):
        tracker.log_run(config={"chunk_size": 512}, metrics={"hit_rate_at_k": 0.8})
        assert (tmp_path / "experiments.jsonl").exists()

    def test_creates_individual_run_file(self, tracker, tmp_path):
        run_id = tracker.log_run(config={}, metrics={})
        assert (tmp_path / f"run_{run_id}.json").exists()

    def test_run_id_format(self, tracker):
        run_id = tracker.log_run(config={}, metrics={})
        assert len(run_id) == 15           # YYYYMMDD_HHMMSS
        assert run_id[8] == "_"
        assert run_id[:8].isdigit()

    def test_tags_persisted(self, tracker):
        tracker.log_run(config={}, metrics={}, tags={"note": "baseline"})
        runs = tracker.load_all_runs()
        assert runs[0]["tags"]["note"] == "baseline"

    def test_config_and_metrics_persisted(self, tracker):
        tracker.log_run(
            config={"chunk_size": 256, "strategy": "hybrid"},
            metrics={"hit_rate_at_k": 0.9, "mrr": 0.75},
        )
        runs = tracker.load_all_runs()
        assert runs[0]["config"]["chunk_size"] == 256
        assert runs[0]["metrics"]["mrr"] == 0.75

    def test_multiple_runs_appended(self, tracker):
        for i in range(3):
            tracker.log_run(config={"i": i}, metrics={"v": float(i)})
        assert len(tracker.load_all_runs()) == 3


class TestLoadAllRuns:
    def test_empty_when_no_runs(self, tracker):
        assert tracker.load_all_runs() == []

    def test_oldest_first_order(self, tracker):
        tracker.log_run(config={"order": 1}, metrics={})
        tracker.log_run(config={"order": 2}, metrics={})
        runs = tracker.load_all_runs()
        assert runs[0]["config"]["order"] == 1
        assert runs[1]["config"]["order"] == 2

    def test_malformed_jsonl_line_is_skipped(self, tracker, tmp_path):
        tracker.log_run(config={}, metrics={"v": 1.0})
        with (tmp_path / "experiments.jsonl").open("a") as fh:
            fh.write("NOT_VALID_JSON\n")
        tracker.log_run(config={}, metrics={"v": 2.0})
        runs = tracker.load_all_runs()
        assert len(runs) == 2  # bad line was silently skipped


class TestGetBestRun:
    def test_returns_highest_metric(self, tracker):
        tracker.log_run(config={}, metrics={"hit_rate_at_k": 0.6})
        tracker.log_run(config={}, metrics={"hit_rate_at_k": 0.9})
        best = tracker.get_best_run("hit_rate_at_k")
        assert best["metrics"]["hit_rate_at_k"] == 0.9

    def test_lower_is_better(self, tracker):
        tracker.log_run(config={}, metrics={"loss": 0.3})
        tracker.log_run(config={}, metrics={"loss": 0.1})
        best = tracker.get_best_run("loss", higher_is_better=False)
        assert best["metrics"]["loss"] == 0.1

    def test_missing_metric_returns_none(self, tracker):
        tracker.log_run(config={}, metrics={"mrr": 0.5})
        assert tracker.get_best_run("hit_rate_at_k") is None

    def test_empty_tracker_returns_none(self, tracker):
        assert tracker.get_best_run("hit_rate_at_k") is None


class TestSummaryTable:
    def test_empty_message_when_no_runs(self, tracker):
        assert "No experiment runs" in tracker.summary_table()

    def test_table_contains_run_id(self, tracker):
        run_id = tracker.log_run(
            config={"chunk_size": 512, "chunking_strategy": "sentence"},
            metrics={"hit_rate_at_k": 0.85, "mrr": 0.70, "k": 5},
        )
        table = tracker.summary_table()
        assert run_id[-8:] in table

    def test_table_is_markdown(self, tracker):
        tracker.log_run(config={"chunk_size": 512}, metrics={"k": 5})
        table = tracker.summary_table()
        assert "|" in table
