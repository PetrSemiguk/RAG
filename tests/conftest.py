"""
Shared pytest configuration.

Tests that require the full ML stack (torch, llama-index, qdrant) declare:

    pytest.importorskip("torch")
    pytest.importorskip("llama_index.core")

at module level.  When those packages are absent (e.g. in CI with
requirements-dev.txt), pytest skips those files cleanly instead of erroring.

The four core test files (test_utils, test_benchmark,
test_experiment_tracker, test_query_logger) have no ML dependencies and
always run.
"""
