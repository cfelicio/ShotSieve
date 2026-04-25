"""Thread-safe registry for managing async background job lifecycles.

Provides a single abstraction for the create → progress → complete/fail/cancel
pattern used by scan, score, and compare job endpoints.
"""
from __future__ import annotations

import threading
import time
from typing import cast
from uuid import uuid4


def _coerce_timestamp(value: object, *, default: float = 0.0) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    return default


def _summary_payload_or_none(value: object) -> dict[str, object] | None:
    if not isinstance(value, dict):
        return None
    return cast(dict[str, object], value)


class JobRegistry:
    """Thread-safe store for async job records.

    Each registry instance manages its own lock, job dict, and eviction
    policy.  Multiple registries can coexist (e.g. one per job type)
    without interfering.

    Typical lifecycle::

        registry = JobRegistry(max_jobs=10)
        job_id = registry.create(initial_progress={...})
        # background thread:
        registry.update_progress(job_id, {...})  # raises InterruptedError if cancelled
        registry.complete(job_id, summary={...})
        # handler:
        payload = registry.status(job_id)        # returns snapshot dict, including summary when completed
    """

    def __init__(self, *, max_jobs: int = 10) -> None:
        self._lock = threading.Lock()
        self._jobs: dict[str, dict[str, object]] = {}
        self._max_jobs = max_jobs

    # ------------------------------------------------------------------
    # Lifecycle methods
    # ------------------------------------------------------------------

    def create(self, *, initial_progress: dict) -> str:
        """Create a new running job record and return job_id.

        Automatically evicts the oldest completed/failed jobs when the
        registry exceeds *max_jobs*.
        """
        job_id = uuid4().hex
        now = time.time()
        with self._lock:
            self._jobs[job_id] = {
                "status": "running",
                "cancelled": False,
                "created_at": now,
                "updated_at": now,
                "finished_at": None,
                "progress": dict(initial_progress),
                "summary": None,
                "error": None,
            }
            self._evict_stale()
        return job_id

    def update_progress(self, job_id: str, progress: dict) -> None:
        """Update progress for a running job.

        Raises :class:`InterruptedError` if the job has been cancelled,
        allowing the worker thread to terminate gracefully.
        """
        with self._lock:
            record = self._jobs.get(job_id)
            if record is None:
                return
            if record.get("cancelled"):
                raise InterruptedError(f"Job {job_id} was cancelled by user.")
            record["progress"] = progress
            record["updated_at"] = time.time()

    def complete(self, job_id: str, *, summary: dict) -> None:
        """Mark job as completed and store the summary payload."""
        now = time.time()
        with self._lock:
            record = self._jobs.get(job_id)
            if record is not None:
                record["status"] = "completed"
                record["summary"] = summary
                record["finished_at"] = now
                record["updated_at"] = now

    def fail(self, job_id: str, *, error: str, progress: dict | None = None) -> None:
        """Mark job as failed and store the error message."""
        now = time.time()
        with self._lock:
            record = self._jobs.get(job_id)
            if record is not None:
                record["status"] = "failed"
                record["error"] = error
                if progress is not None:
                    record["progress"] = dict(progress)
                record["finished_at"] = now
                record["updated_at"] = now

    def cancel(self, job_id: str) -> bool:
        """Request cancellation.  Returns True if the job was running."""
        with self._lock:
            record = self._jobs.get(job_id)
            if record is not None and record.get("status") == "running":
                record["cancelled"] = True
                return True
            return False

    def is_cancelled(self, job_id: str) -> bool:
        """Check whether a job has been flagged for cancellation."""
        with self._lock:
            record = self._jobs.get(job_id)
            return bool(record is not None and record.get("cancelled"))

    # ------------------------------------------------------------------
    # Query methods
    # ------------------------------------------------------------------

    def status(self, job_id: str) -> dict[str, object] | None:
        """Build a status payload for the given job.

        Returns ``None`` if the job_id is unknown.  The payload includes:
        ``job_id``, ``status``, ``progress``, ``elapsed_seconds``,
        and optionally ``summary`` and ``error``. Completed summaries remain
        available until eviction so result endpoints can be retried safely.
        """
        with self._lock:
            record = self._jobs.get(job_id)
            if record is None:
                return None
            snapshot = dict(record)

        created_at = _coerce_timestamp(snapshot.get("created_at"))
        finished_at = snapshot.get("finished_at")
        now = _coerce_timestamp(finished_at, default=time.time())
        elapsed_seconds = max(0.0, round(now - created_at, 4))

        payload: dict[str, object] = {
            "job_id": job_id,
            "status": snapshot.get("status", "running"),
            "progress": snapshot.get("progress"),
            "elapsed_seconds": elapsed_seconds,
        }
        if snapshot.get("summary") is not None:
            payload["summary"] = snapshot["summary"]
        if snapshot.get("error"):
            payload["error"] = snapshot["error"]
        return payload

    def pop_result(self, job_id: str) -> dict[str, object] | None:
        """Return the summary for a completed job and remove it from the registry.

        Returns ``None`` if the job is not found or not yet completed.

        This helper is intended for explicit acknowledgement/cleanup flows; the
        web result endpoints keep completed summaries retryable until eviction.
        """
        with self._lock:
            record = self._jobs.get(job_id)
            if record is None or record.get("status") != "completed":
                return None
            summary = _summary_payload_or_none(record.get("summary"))
            self._jobs.pop(job_id, None)
        return summary

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _evict_stale(self) -> None:
        """Remove oldest finished jobs when over capacity.  Caller must hold ``_lock``."""
        if len(self._jobs) <= self._max_jobs:
            return
        finished_ids = sorted(
            (jid for jid, rec in self._jobs.items() if rec.get("status") in {"completed", "failed"}),
            key=lambda jid: _coerce_timestamp(self._jobs[jid].get("finished_at")),
        )
        for stale_id in finished_ids[: len(self._jobs) - self._max_jobs]:
            self._jobs.pop(stale_id, None)
