from __future__ import annotations

import json
import logging
import sqlite3

from shotsieve.performance import PERFORMANCE_LOGGING_ENV, explain_query_plan, log_duration


def test_log_duration_is_silent_until_explicitly_enabled(monkeypatch, caplog) -> None:
    monkeypatch.delenv(PERFORMANCE_LOGGING_ENV, raising=False)
    caplog.set_level(logging.DEBUG, logger="shotsieve.performance")

    elapsed_ms = log_duration("review.list", 0.0, scope="root", result_count=60)

    assert elapsed_ms >= 0
    assert not caplog.records


def test_log_duration_emits_structured_aggregate_metrics_when_enabled(monkeypatch, caplog) -> None:
    monkeypatch.setenv(PERFORMANCE_LOGGING_ENV, "true")
    caplog.set_level(logging.DEBUG, logger="shotsieve.performance")

    log_duration("review.list", 0.0, scope="root", result_count=60)

    assert len(caplog.records) == 1
    payload = json.loads(caplog.records[0].message)
    assert payload["event"] == "shotsieve_performance"
    assert payload["operation"] == "review.list"
    assert payload["scope"] == "root"
    assert payload["result_count"] == 60
    assert isinstance(payload["elapsed_ms"], float)


def test_explain_query_plan_returns_sqlite_details_without_executing_statement() -> None:
    connection = sqlite3.connect(":memory:")
    try:
        connection.execute("CREATE TABLE samples (id INTEGER PRIMARY KEY, value TEXT NOT NULL)")
        connection.execute("CREATE INDEX idx_samples_value ON samples(value)")

        plan = explain_query_plan(connection, "SELECT id FROM samples WHERE value = ?", ("example",))
    finally:
        connection.close()

    assert plan
    assert any("idx_samples_value" in detail for detail in plan)
