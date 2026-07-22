"""Opt-in performance diagnostics for local development and benchmark runs."""
from __future__ import annotations

import json
import logging
import os
import sqlite3
from collections.abc import Mapping, Sequence
from time import perf_counter as _perf_counter
from typing import Any


PERFORMANCE_LOGGING_ENV = "SHOTSIEVE_PERFORMANCE_LOGGING"
_LOG = logging.getLogger(__name__)


def performance_logging_enabled(*, environ: Mapping[str, str] | None = None) -> bool:
    """Return whether structured performance logging is explicitly enabled."""
    source = os.environ if environ is None else environ
    return source.get(PERFORMANCE_LOGGING_ENV, "").strip().casefold() in {"1", "true", "yes", "on"}


def monotonic_seconds() -> float:
    """Return the monotonic clock used to calculate diagnostic durations."""
    return _perf_counter()


def log_duration(
    operation: str,
    started_at: float,
    /,
    **metrics: Any,
) -> float:
    """Log an opt-in structured duration event and return elapsed milliseconds.

    Metrics must be small, non-sensitive aggregate values. Callers must not pass
    file paths, search terms, or other user-library content.
    """
    elapsed_ms = round(max(0.0, monotonic_seconds() - started_at) * 1000, 3)
    if not performance_logging_enabled():
        return elapsed_ms

    payload = {
        "event": "shotsieve_performance",
        "operation": operation,
        "elapsed_ms": elapsed_ms,
        **metrics,
    }
    _LOG.debug("%s", json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str))
    return elapsed_ms


def explain_query_plan(
    connection: sqlite3.Connection,
    sql: str,
    params: Sequence[object] = (),
) -> list[str]:
    """Return SQLite's developer-facing query-plan details without running SQL."""
    rows = connection.execute(f"EXPLAIN QUERY PLAN {sql}", tuple(params)).fetchall()
    return [str(row[3]) for row in rows]
