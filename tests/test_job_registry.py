from __future__ import annotations

from shotsieve.job_registry import JobRegistry



def test_status_tolerates_invalid_timestamp_values() -> None:
    registry = JobRegistry(max_jobs=10)
    job_id = registry.create(initial_progress={"files_processed": 0})

    with registry._lock:
        registry._jobs[job_id]["created_at"] = object()
        registry._jobs[job_id]["finished_at"] = object()

    payload = registry.status(job_id)

    assert payload is not None
    assert payload["status"] == "running"
    assert isinstance(payload["elapsed_seconds"], float)
    assert payload["elapsed_seconds"] >= 0.0



def test_pop_result_rejects_non_dict_summary_payloads() -> None:
    registry = JobRegistry(max_jobs=10)
    job_id = registry.create(initial_progress={"files_processed": 0})

    with registry._lock:
        registry._jobs[job_id]["status"] = "completed"
        registry._jobs[job_id]["summary"] = object()

    result = registry.pop_result(job_id)

    assert result is None



def test_evict_stale_tolerates_invalid_finished_timestamps() -> None:
    registry = JobRegistry(max_jobs=1)
    older_job_id = registry.create(initial_progress={"files_processed": 0})
    newer_job_id = registry.create(initial_progress={"files_processed": 0})

    with registry._lock:
        registry._jobs[older_job_id]["status"] = "completed"
        registry._jobs[older_job_id]["finished_at"] = object()
        registry._jobs[newer_job_id]["status"] = "completed"
        registry._jobs[newer_job_id]["finished_at"] = 1.0
        registry._evict_stale()

    assert len(registry._jobs) == 1
    assert newer_job_id in registry._jobs
