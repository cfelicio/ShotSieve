from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
import time
from typing import Callable, Protocol, Sequence, runtime_checkable

from shotsieve.config import ALL_PREVIEWABLE_EXTENSIONS, DEFAULT_RAW_PREVIEW_MODE, PIL_ANALYSIS_EXTENSIONS, PREVIEW_PRIORITY_EXTENSIONS
from shotsieve.db import roots_path_filter, set_preview_cache_root
from shotsieve.image_conversion import IMAGE_CONVERSION_VERSION
from shotsieve.performance import log_duration, monotonic_seconds
from shotsieve.learned_iqa import DEFAULT_BATCH_SIZE, DEFAULT_MODEL_NAME, LearnedIqaBackend, LearnedScoreResult, build_learned_backend, release_learned_backend, recommended_batch_size, recommended_cpu_workers, detect_hardware_capabilities, resolve_learned_model_version, validate_model_name
from shotsieve.model_assets import attach_model_diagnostic
from shotsieve.preview import PreviewResult, generate_previews_parallel
from shotsieve.scanner import utc_now


log = logging.getLogger(__name__)


COMPARE_MAX_ROWS = 10_000


def _detect_vram_lazy() -> int | None:
    """Return cached VRAM in MB (detected once per process via learned_iqa)."""
    value = detect_hardware_capabilities().get("vram_mb")
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        try:
            return int(stripped)
        except ValueError:
            return None
    return None


def _coerce_int(value: object) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        return int(value)
    raise TypeError(f"Expected int-like value, got {type(value).__name__}")


@runtime_checkable
class _RowLike(Protocol):
    def __getitem__(self, key: str, /) -> object: ...


def _row_like(row: object) -> _RowLike:
    if not isinstance(row, _RowLike):
        raise TypeError(f"Expected row-like object, got {type(row).__name__}")
    return row


def _row_int(row: object, key: str) -> int:
    return _coerce_int(_row_like(row)[key])


def _row_text(row: object, key: str) -> str:
    return str(_row_like(row)[key])


def _default_preview_workers(resource_profile: str | None = None) -> int:
    """Scale preview workers with available CPU cores and system RAM.

    Uses :func:`recommended_cpu_workers` which adapts to both core count
    and available RAM — preventing OOM on machines with many cores but
    limited memory (e.g. 64-core / 4 GB cloud VMs).
    """
    return recommended_cpu_workers(resource_profile)


LEARNED_ONLY_PRESET = "learned-only"


def _db_model_version(model_version: str | None) -> str:
    if not model_version:
        return ""

    normalized = str(model_version).strip()
    if normalized.startswith("learned:"):
        return normalized
    return f"learned:{normalized}"


def _has_ready_preview(row) -> bool:
    return (
        bool(row["preview_path"])
        and row["preview_status"] == "ready"
        and _optional_row_value(row, "preview_conversion_version", IMAGE_CONVERSION_VERSION)
        == IMAGE_CONVERSION_VERSION
    )


def _has_stale_preview_conversion(row) -> bool:
    return (
        bool(row["preview_path"])
        and row["preview_status"] == "ready"
        and _optional_row_value(row, "preview_conversion_version", IMAGE_CONVERSION_VERSION)
        != IMAGE_CONVERSION_VERSION
    )


def _optional_row_value(row, key: str, default=None):
    try:
        return row[key]
    except (KeyError, IndexError):
        return default


def _should_prioritize_generated_preview_for_review(source_path: Path) -> bool:
    return source_path.suffix.casefold() in PREVIEW_PRIORITY_EXTENSIONS


def _is_failed_learned_result(result: LearnedScoreResult) -> bool:
    return result.failed


@dataclass(slots=True)
class ScoreSummary:
    rows_loaded: int = 0
    files_considered: int = 0
    files_scored: int = 0
    learned_scored: int = 0
    files_skipped: int = 0
    files_failed: int = 0


@dataclass(slots=True)
class ModelComparisonSummary:
    model_names: list[str] = field(default_factory=list)
    rows: list[dict[str, object]] = field(default_factory=list)
    compare_failures: list[dict[str, object]] = field(default_factory=list)
    requested_rows_total: int = 0
    processed_rows_total: int = 0
    truncated: bool = False
    max_rows: int = COMPARE_MAX_ROWS
    files_considered: int = 0
    files_compared: int = 0
    files_skipped: int = 0
    files_failed: int = 0
    elapsed_seconds: float = 0.0
    model_timings_seconds: dict[str, float] = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class AnalysisProgress:
    model_name: str | None
    model_index: int
    model_count: int
    files_processed: int
    files_total: int
    phase: str = "scoring"


@dataclass(slots=True, frozen=True)
class PreparedAnalysisCandidate:
    row: object
    analysis_path: Path


@dataclass(slots=True, frozen=True)
class GeneratedPreviewCandidate:
    row: object
    preview_result: PreviewResult


@dataclass(slots=True)
class PreparedAnalysisBatch:
    analysis_candidates: list[PreparedAnalysisCandidate] = field(default_factory=list)
    generated_preview_results: list[GeneratedPreviewCandidate] = field(default_factory=list)
    unresolved_preview_results: list[GeneratedPreviewCandidate] = field(default_factory=list)
    unavailable_rows: list[object] = field(default_factory=list)
    has_ready_generated_preview: bool = False


@dataclass(slots=True)
class _ScorePlan:
    """Rows and counters produced before preview/model work begins."""

    rows_for_scoring: list[object] = field(default_factory=list)
    files_considered: int = 0


@dataclass(slots=True)
class _ScorePreviewOutcome:
    """Persistence outcomes that belong to the pre-inference phase."""

    files_skipped: int = 0
    files_failed: int = 0


@dataclass(slots=True)
class _ScoreBatchOutcome:
    """Counters accumulated while one learned backend scores its batches."""

    files_scored: int = 0
    learned_scored: int = 0
    files_failed: int = 0


def _build_score_plan(
    rows,
    *,
    force: bool,
    selected_backend: str,
    version_resolver: Callable[[str], str | None] | None,
    custom_backend_factory: bool,
) -> _ScorePlan:
    """Select rows needing work and apply same-backend version invalidation.

    The resolver is deliberately supplied by :func:`score_files`. That keeps
    the default model-version probe and the custom test/integration seam at the
    public workflow boundary while making row selection independently testable.
    """
    plan = _ScorePlan()
    version_check_candidates: list[object] = []

    for row in rows:
        if needs_score_update(row, force=force, learned_backend_name=selected_backend):
            plan.rows_for_scoring.append(row)
            continue

        if row["existing_score_id"] is not None and row["learned_backend"] == selected_backend:
            version_check_candidates.append(row)

    if version_check_candidates:
        if version_resolver is not None:
            try:
                resolved_model_version = version_resolver(selected_backend)
            except Exception as exc:
                log.warning(
                    "Falling back to rescoring same-backend cached rows for '%s' because version probing failed: %s",
                    selected_backend,
                    exc,
                )
                plan.rows_for_scoring.extend(version_check_candidates)
            else:
                if not resolved_model_version:
                    log.warning(
                        "Falling back to rescoring same-backend cached rows for '%s' because version probing returned no version.",
                        selected_backend,
                    )
                    plan.rows_for_scoring.extend(version_check_candidates)
                else:
                    expected_model_version = _db_model_version(resolved_model_version)
                    plan.rows_for_scoring.extend(
                        row
                        for row in version_check_candidates
                        if needs_score_update(
                            row,
                            force=False,
                            learned_backend_name=selected_backend,
                            expected_model_version=expected_model_version,
                        )
                    )
        elif custom_backend_factory:
            log.warning(
                "Custom learned_backend_factory supplied without learned_model_version_resolver; "
                "same-backend model-version invalidation is skipped for '%s'.",
                selected_backend,
            )

    plan.files_considered = len(plan.rows_for_scoring)
    return plan


def _drop_stale_score_row(connection, row) -> None:
    if row["existing_score_id"] is not None:
        delete_score_row(connection, file_id=_row_int(row, "id"))


def _record_analysis_outcome(
    connection,
    row,
    *,
    status: str,
    error: str | None = None,
) -> None:
    connection.execute(
        """
        UPDATE files
        SET analysis_status = ?, analysis_error = ?, last_analysis_time = ?
        WHERE id = ?
        """,
        (status, error, utc_now(), _row_int(row, "id")),
    )


def _persist_generated_preview_results(
    connection,
    prepared: PreparedAnalysisBatch,
    *,
    preview_dir: Path | None,
) -> None:
    """Persist generated previews and register a successful cache location."""
    if prepared.has_ready_generated_preview and preview_dir is not None:
        try:
            set_preview_cache_root(connection, preview_dir)
        except ValueError:
            pass

    for generated_preview in prepared.generated_preview_results:
        persist_generated_preview(
            connection,
            row_id=_row_int(generated_preview.row, "id"),
            preview_result=generated_preview.preview_result,
        )


def _persist_score_preview_outcomes(
    connection,
    prepared: PreparedAnalysisBatch,
    *,
    preview_dir: Path | None,
) -> _ScorePreviewOutcome:
    """Persist preview results and mark rows that cannot reach inference."""
    _persist_generated_preview_results(connection, prepared, preview_dir=preview_dir)
    outcome = _ScorePreviewOutcome()

    for row in prepared.unavailable_rows:
        _drop_stale_score_row(connection, row)
        _record_analysis_outcome(
            connection,
            row,
            status="skipped",
            error="No usable source image or ready preview is available for analysis.",
        )
        outcome.files_skipped += 1

    for unresolved_preview in prepared.unresolved_preview_results:
        _drop_stale_score_row(connection, unresolved_preview.row)
        if unresolved_preview.preview_result.status == "failed":
            _record_analysis_outcome(
                connection,
                unresolved_preview.row,
                status="failed",
                error=unresolved_preview.preview_result.error_text
                or "Preview generation failed before analysis.",
            )
            outcome.files_failed += 1
        else:
            _record_analysis_outcome(
                connection,
                unresolved_preview.row,
                status="skipped",
                error="A preview was unavailable for analysis.",
            )
            outcome.files_skipped += 1

    return outcome


def _run_score_batches(
    connection,
    pending_learned: list[tuple[object, Path]],
    *,
    backend: LearnedIqaBackend,
    learned_batch_size: int,
    resource_profile: str | None,
    progress_callback: Callable[[AnalysisProgress], None] | None,
) -> _ScoreBatchOutcome:
    """Run inference batches and persist each learned score outcome."""
    outcome = _ScoreBatchOutcome()
    files_total = len(pending_learned)
    if progress_callback is not None:
        progress_callback(
            AnalysisProgress(
                model_name=backend.name,
                model_index=1,
                model_count=1,
                files_processed=0,
                files_total=files_total,
                phase="scoring",
            )
        )

    # Keep scoring updates frequent enough to avoid long visible stalls at 0/N.
    progress_chunk_size = max(learned_batch_size, 12)
    for chunk_start in range(0, files_total, progress_chunk_size):
        chunk_end = min(chunk_start + progress_chunk_size, files_total)
        chunk = pending_learned[chunk_start:chunk_end]
        learned_results = backend.score_paths(
            [analysis_path for _, analysis_path in chunk],
            batch_size=learned_batch_size,
            resource_profile=resource_profile,
        )

        for (row, _), learned_result in zip(chunk, learned_results, strict=True):
            if _is_failed_learned_result(learned_result):
                delete_score_row(connection, file_id=_row_int(row, "id"))
                _record_analysis_outcome(
                    connection,
                    row,
                    status="failed",
                    error=learned_result.error or "The learned quality model returned no score.",
                )
                outcome.files_failed += 1
                log.warning(
                    "Failed to score %s with backend %s: %s",
                    _row_text(row, "path"),
                    backend.name,
                    learned_result.error or "unknown learned IQA failure",
                )
                continue

            upsert_score_row(
                connection,
                row,
                learned=learned_result,
                learned_backend=backend.name,
                learned_model_version=backend.model_version,
            )
            _record_analysis_outcome(connection, row, status="ready")
            outcome.files_scored += 1
            outcome.learned_scored += 1

        if progress_callback is not None:
            progress_callback(
                AnalysisProgress(
                    model_name=backend.name,
                    model_index=1,
                    model_count=1,
                    files_processed=chunk_end,
                    files_total=files_total,
                    phase="scoring",
                )
            )

    return outcome


def _prepare_analysis_candidates(
    rows,
    *,
    preview_dir: Path | None,
    preview_workers: int | None,
    raw_preview_mode: str,
    resource_profile: str | None,
    preview_progress_callback: Callable[[int, int], None] | None,
    preview_start_callback: Callable[[int], None] | None = None,
) -> PreparedAnalysisBatch:
    prepared = PreparedAnalysisBatch()
    preview_candidates: list[tuple[object, Path]] = []

    for row in rows:
        source_path = Path(row["path"])
        can_generate_preview = (
            preview_dir is not None
            and source_path.exists()
            and source_path.suffix.casefold() in ALL_PREVIEWABLE_EXTENSIONS
        )

        if (
            can_generate_preview
            and not _has_ready_preview(row)
            and (
                _should_prioritize_generated_preview_for_review(source_path)
                or _has_stale_preview_conversion(row)
            )
        ):
            preview_candidates.append((row, source_path))
            continue

        analysis_path = select_analysis_path(
            row["path"],
            row["preview_path"],
            row["preview_status"],
            preview_conversion_version=_optional_row_value(row, "preview_conversion_version"),
        )
        if analysis_path is not None:
            prepared.analysis_candidates.append(
                PreparedAnalysisCandidate(row=row, analysis_path=analysis_path)
            )
            continue

        if can_generate_preview:
            preview_candidates.append((row, source_path))
            continue

        prepared.unavailable_rows.append(row)

    if not preview_candidates:
        return prepared

    if preview_dir is None:
        return prepared

    pc_rows = [row for row, _ in preview_candidates]
    pc_paths = [path for _, path in preview_candidates]
    if preview_start_callback is not None:
        preview_start_callback(len(pc_paths))

    effective_workers = preview_workers or _default_preview_workers(resource_profile)
    preview_results = generate_previews_parallel(
        pc_paths,
        preview_dir,
        max_workers=effective_workers,
        progress_callback=preview_progress_callback,
        raw_preview_mode=raw_preview_mode,
    )

    for row, preview_result in zip(pc_rows, preview_results, strict=True):
        generated_preview = GeneratedPreviewCandidate(row=row, preview_result=preview_result)
        prepared.generated_preview_results.append(generated_preview)

        if preview_result.status == "ready" and preview_result.path:
            prepared.has_ready_generated_preview = True
            resolved_preview_path = Path(preview_result.path)
            if resolved_preview_path.exists():
                prepared.analysis_candidates.append(
                    PreparedAnalysisCandidate(row=row, analysis_path=resolved_preview_path)
                )
                continue

        fallback_analysis_path = select_analysis_path(_row_text(row, "path"), None, None)
        if fallback_analysis_path is not None:
            prepared.analysis_candidates.append(
                PreparedAnalysisCandidate(row=row, analysis_path=fallback_analysis_path)
            )
            continue

        prepared.unresolved_preview_results.append(generated_preview)

    return prepared


def score_files(
    connection,
    *,
    limit: int | None = None,
    offset: int = 0,
    raw_root: str | Sequence[str] | Sequence[Path] | None = None,
    force: bool = False,
    learned_backend_name: str | None = None,
    learned_device: str | None = None,
    learned_batch_size: int = DEFAULT_BATCH_SIZE,
    preview_dir: Path | None = None,

    preview_workers: int | None = None,
    raw_preview_mode: str = DEFAULT_RAW_PREVIEW_MODE,
    learned_backend_factory: Callable[[str], LearnedIqaBackend] | None = None,
    learned_model_version_resolver: Callable[[str], str | None] | None = None,
    progress_callback: Callable[[AnalysisProgress], None] | None = None,
    resource_profile: str | None = None,
) -> ScoreSummary:
    """Score catalog rows while preserving the public progress/lifecycle contract."""
    summary = ScoreSummary()
    selected_backend = validate_model_name(learned_backend_name or DEFAULT_MODEL_NAME)

    rows = fetch_score_rows(connection, raw_root=raw_root, limit=limit, offset=offset)
    summary.rows_loaded = len(rows)
    if learned_backend_factory is None:
        def factory(model_name: str) -> LearnedIqaBackend:
            return build_learned_backend(model_name, device=learned_device)
    else:
        factory = learned_backend_factory

    version_resolver = learned_model_version_resolver
    if version_resolver is None and learned_backend_factory is None:
        def resolve_model_version(model_name: str) -> str | None:
            try:
                return resolve_learned_model_version(model_name, device=learned_device)
            except Exception as exc:
                attach_model_diagnostic(
                    exc,
                    phase="resolving_model_version",
                    model_name=model_name,
                    requested_runtime=learned_device,
                )
                raise

        version_resolver = resolve_model_version

    plan = _build_score_plan(
        rows,
        force=force,
        selected_backend=selected_backend,
        version_resolver=version_resolver,
        custom_backend_factory=learned_backend_factory is not None,
    )
    summary.files_considered = plan.files_considered

    def _score_preview_start(total: int) -> None:
        if progress_callback is not None:
            progress_callback(
                AnalysisProgress(
                    model_name=None,
                    model_index=0,
                    model_count=1,
                    files_processed=0,
                    files_total=total,
                    phase="generating_previews",
                )
            )

    def _score_preview_progress(completed: int, total: int) -> None:
        if progress_callback is not None:
            progress_callback(
                AnalysisProgress(
                    model_name=None,
                    model_index=0,
                    model_count=1,
                    files_processed=completed,
                    files_total=total,
                    phase="generating_previews",
                )
            )

    prepared = _prepare_analysis_candidates(
        plan.rows_for_scoring,
        preview_dir=preview_dir,
        preview_workers=preview_workers,
        raw_preview_mode=raw_preview_mode,
        resource_profile=resource_profile,
        preview_progress_callback=_score_preview_progress,
        preview_start_callback=_score_preview_start,
    )

    pending_learned = [
        (candidate.row, candidate.analysis_path)
        for candidate in prepared.analysis_candidates
    ]

    preview_outcome = _persist_score_preview_outcomes(
        connection,
        prepared,
        preview_dir=preview_dir,
    )
    summary.files_skipped += preview_outcome.files_skipped
    summary.files_failed += preview_outcome.files_failed

    if not pending_learned:
        return summary

    files_total = len(pending_learned)
    if progress_callback is not None:
        progress_callback(
            AnalysisProgress(
                model_name=selected_backend,
                model_index=1,
                model_count=1,
                files_processed=0,
                files_total=files_total,
                phase="loading",
            )
        )

    backend = None
    try:
        backend = factory(selected_backend)
        outcome = _run_score_batches(
            connection,
            pending_learned,
            backend=backend,
            learned_batch_size=learned_batch_size,
            resource_profile=resource_profile,
            progress_callback=progress_callback,
        )
        summary.files_scored += outcome.files_scored
        summary.learned_scored += outcome.learned_scored
        summary.files_failed += outcome.files_failed
    except Exception as exc:
        attach_model_diagnostic(
            exc,
            phase="scoring",
            model_name=selected_backend,
            requested_runtime=learned_device,
            actual_runtime=getattr(backend, "runtime", None),
        )
        raise
    finally:
        if backend is not None:
            release_learned_backend(backend)

    return summary




def persist_generated_preview(connection, *, row_id: int, preview_result: PreviewResult) -> None:
    connection.execute(
        """
        UPDATE files
        SET preview_path = ?,
            preview_status = ?,
            preview_conversion_version = ?,
            width = ?,
            height = ?,
            capture_time = ?,
            last_scan_time = ?,
            last_error = ?
        WHERE id = ?
        """,
        (
            preview_result.path,
            preview_result.status,
            IMAGE_CONVERSION_VERSION if preview_result.status == "ready" else None,
            preview_result.width,
            preview_result.height,
            preview_result.capture_time,
            utc_now(),
            preview_result.error_text,
            row_id,
        ),
    )


def _compare_failure_detail(
    row,
    *,
    reason: str | None,
    stage: str,
) -> dict[str, object]:
    reason_text = (reason or "").strip() or "Comparison preparation failed"
    return {
        "file_id": int(row["id"]),
        "path": str(row["path"]),
        "reason": reason_text,
        "stage": stage,
    }


def _accumulate_comparison_result(
    summary: ModelComparisonSummary,
    row: dict[str, object],
    *,
    backend_name: str,
    learned_result: LearnedScoreResult,
    failed_file_ids: set[int],
) -> None:
    """Add one model result to a stable side-by-side comparison row."""
    prefix = backend_name
    if _is_failed_learned_result(learned_result):
        row[f"{prefix}_score"] = None
        row[f"{prefix}_confidence"] = None
        row[f"{prefix}_raw"] = None
        row[f"{prefix}_error"] = learned_result.error
        failed_file_ids.add(_coerce_int(row["file_id"]))
        summary.files_failed = len(failed_file_ids)
        return

    row[f"{prefix}_score"] = learned_result.normalized_score
    row[f"{prefix}_confidence"] = learned_result.confidence
    row[f"{prefix}_raw"] = learned_result.raw_score
    row[f"{prefix}_error"] = None


def _run_comparison_model(
    summary: ModelComparisonSummary,
    candidate_rows: list[dict[str, object]],
    analysis_paths: list[Path],
    *,
    model_name: str,
    model_index: int,
    model_count: int,
    learned_device: str | None,
    learned_batch_size: int,
    compare_chunk_size: int | None,
    learned_backend_factory: Callable[[str], LearnedIqaBackend],
    progress_callback: Callable[[AnalysisProgress], None] | None,
    release_backends: bool,
    resource_profile: str | None,
    failed_file_ids: set[int],
) -> None:
    """Load, run, and release one comparison model."""
    if progress_callback is not None:
        progress_callback(
            AnalysisProgress(
                model_name=model_name,
                model_index=model_index,
                model_count=model_count,
                files_processed=0,
                files_total=len(candidate_rows),
                phase="loading",
            )
        )

    model_started_at = time.perf_counter()
    backend = None

    # Use per-model optimal batch size instead of a single global size. This
    # prevents lightweight models from being throttled by a heavier model.
    model_batch_size = recommended_batch_size(
        model_name,
        vram_mb=_detect_vram_lazy(),
        resource_profile=resource_profile,
    )
    # An explicit caller batch size remains an upper bound. The default means
    # "use the model recommendation" rather than "use the default literally".
    effective_batch_size = (
        min(max(1, learned_batch_size), model_batch_size)
        if learned_batch_size != DEFAULT_BATCH_SIZE
        else model_batch_size
    )
    effective_compare_chunk_size = max(effective_batch_size, effective_batch_size * 16)
    if compare_chunk_size is not None:
        effective_compare_chunk_size = max(1, compare_chunk_size)

    try:
        backend = learned_backend_factory(model_name)
        if progress_callback is not None:
            progress_callback(
                AnalysisProgress(
                    model_name=backend.name,
                    model_index=model_index,
                    model_count=model_count,
                    files_processed=0,
                    files_total=len(candidate_rows),
                    phase="scoring",
                )
            )

        for chunk_start in range(0, len(candidate_rows), effective_compare_chunk_size):
            chunk_end = min(chunk_start + effective_compare_chunk_size, len(candidate_rows))
            learned_results = backend.score_paths(
                analysis_paths[chunk_start:chunk_end],
                batch_size=effective_batch_size,
                resource_profile=resource_profile,
            )
            for row, learned_result in zip(
                candidate_rows[chunk_start:chunk_end],
                learned_results,
                strict=True,
            ):
                _accumulate_comparison_result(
                    summary,
                    row,
                    backend_name=backend.name,
                    learned_result=learned_result,
                    failed_file_ids=failed_file_ids,
                )

            if progress_callback is not None:
                progress_callback(
                    AnalysisProgress(
                        model_name=backend.name,
                        model_index=model_index,
                        model_count=model_count,
                        files_processed=chunk_end,
                        files_total=len(candidate_rows),
                        phase="scoring",
                    )
                )

        summary.model_timings_seconds[backend.name] = round(
            time.perf_counter() - model_started_at,
            4,
        )
    except Exception as exc:
        attach_model_diagnostic(
            exc,
            phase="comparing",
            model_name=model_name,
            requested_runtime=learned_device,
            actual_runtime=getattr(backend, "runtime", None),
        )
        raise
    finally:
        if release_backends and backend is not None:
            release_learned_backend(backend)


def _run_comparison_models(
    summary: ModelComparisonSummary,
    candidate_rows: list[dict[str, object]],
    analysis_paths: list[Path],
    *,
    learned_device: str | None,
    learned_batch_size: int,
    compare_chunk_size: int | None,
    learned_backend_factory: Callable[[str], LearnedIqaBackend],
    progress_callback: Callable[[AnalysisProgress], None] | None,
    release_backends: bool,
    resource_profile: str | None,
    failed_file_ids: set[int],
) -> None:
    """Execute the comparison model loop against shared prepared inputs."""
    model_count = len(summary.model_names)
    for model_index, model_name in enumerate(summary.model_names, start=1):
        _run_comparison_model(
            summary,
            candidate_rows,
            analysis_paths,
            model_name=model_name,
            model_index=model_index,
            model_count=model_count,
            learned_device=learned_device,
            learned_batch_size=learned_batch_size,
            compare_chunk_size=compare_chunk_size,
            learned_backend_factory=learned_backend_factory,
            progress_callback=progress_callback,
            release_backends=release_backends,
            resource_profile=resource_profile,
            failed_file_ids=failed_file_ids,
        )


def compare_learned_models(
    connection,
    *,
    model_names: list[str],
    limit: int | None = None,
    offset: int = 0,
    raw_root: str | Sequence[str] | Sequence[Path] | None = None,
    learned_device: str | None = None,

    learned_batch_size: int = DEFAULT_BATCH_SIZE,
    learned_backend_factory: Callable[[str], LearnedIqaBackend] | None = None,
    compare_chunk_size: int | None = None,
    progress_callback: Callable[[AnalysisProgress], None] | None = None,
    release_backends: bool = True,
    preview_dir: Path | None = None,
    preview_workers: int | None = None,
    raw_preview_mode: str = DEFAULT_RAW_PREVIEW_MODE,
    resource_profile: str | None = None,
) -> ModelComparisonSummary:
    started_at = time.perf_counter()
    normalized_models = [
        validate_model_name(model_name)
        for model_name in model_names
        if model_name.strip()
    ]
    unique_models = list(dict.fromkeys(normalized_models))
    summary = ModelComparisonSummary(model_names=unique_models)

    matching_rows_total = count_score_rows(connection, raw_root=raw_root)
    available_rows_total = max(0, matching_rows_total - max(0, offset))
    requested_rows_total = available_rows_total if limit is None else min(available_rows_total, limit)
    effective_limit = min(requested_rows_total, COMPARE_MAX_ROWS)
    summary.requested_rows_total = requested_rows_total
    summary.max_rows = COMPARE_MAX_ROWS

    rows = fetch_score_rows(connection, raw_root=raw_root, limit=effective_limit, offset=offset)
    summary.processed_rows_total = len(rows)
    summary.truncated = summary.processed_rows_total < summary.requested_rows_total
    summary.files_considered = len(rows)
    failed_file_ids: set[int] = set()

    candidate_rows: list[dict[str, object]] = []
    analysis_paths: list[Path] = []

    def _compare_preview_progress(completed: int, total: int) -> None:
        if progress_callback is not None:
            progress_callback(
                AnalysisProgress(
                    model_name=None,
                    model_index=0,
                    model_count=len(unique_models),
                    files_processed=completed,
                    files_total=total,
                    phase="generating_previews",
                )
            )

    def _compare_preview_start(total: int) -> None:
        if progress_callback is not None:
            progress_callback(
                AnalysisProgress(
                    model_name=None,
                    model_index=0,
                    model_count=len(unique_models),
                    files_processed=0,
                    files_total=total,
                    phase="generating_previews",
                )
            )

    prepared = _prepare_analysis_candidates(
        rows,
        preview_dir=preview_dir,
        preview_workers=preview_workers,
        raw_preview_mode=raw_preview_mode,
        resource_profile=resource_profile,
        preview_progress_callback=_compare_preview_progress,
        preview_start_callback=_compare_preview_start,
    )

    for candidate in prepared.analysis_candidates:
        candidate_rows.append({"file_id": _row_int(candidate.row, "id"), "path": _row_text(candidate.row, "path")})
        analysis_paths.append(candidate.analysis_path)

    _persist_generated_preview_results(connection, prepared, preview_dir=preview_dir)

    summary.files_skipped += len(prepared.unavailable_rows)

    for unresolved_preview in prepared.unresolved_preview_results:
        if unresolved_preview.preview_result.status == "failed":
            failed_file_ids.add(_row_int(unresolved_preview.row, "id"))
            summary.compare_failures.append(
                _compare_failure_detail(
                    unresolved_preview.row,
                    reason=unresolved_preview.preview_result.error_text,
                    stage="preview_generation",
                )
            )
            summary.files_failed = len(failed_file_ids)
        else:
            summary.files_skipped += 1

    summary.files_compared = len(candidate_rows)
    if not candidate_rows or not unique_models:
        summary.rows = candidate_rows
        summary.elapsed_seconds = round(time.perf_counter() - started_at, 4)
        return summary

    if learned_backend_factory is None:
        def factory(model_name: str) -> LearnedIqaBackend:
            return build_learned_backend(model_name, device=learned_device)
    else:
        factory = learned_backend_factory
    _run_comparison_models(
        summary,
        candidate_rows,
        analysis_paths,
        learned_device=learned_device,
        learned_batch_size=learned_batch_size,
        compare_chunk_size=compare_chunk_size,
        learned_backend_factory=factory,
        progress_callback=progress_callback,
        release_backends=release_backends,
        resource_profile=resource_profile,
        failed_file_ids=failed_file_ids,
    )

    summary.rows = candidate_rows
    summary.elapsed_seconds = round(time.perf_counter() - started_at, 4)
    return summary


def _resolve_roots_argument(raw_root: str | Sequence[str] | Sequence[Path] | None) -> list[Path]:
    if not raw_root:
        return []
    if isinstance(raw_root, str):
        paths_str = [p.strip() for p in raw_root.split("|") if p.strip()]
        return [Path(p).expanduser().resolve() for p in paths_str]
    elif isinstance(raw_root, Path):
        return [raw_root.expanduser().resolve()]
    else:
        return [Path(p).expanduser().resolve() for p in raw_root if p]


def fetch_score_rows(connection, *, raw_root: str | Sequence[str] | Sequence[Path] | None, limit: int | None, offset: int = 0):
    started_at = monotonic_seconds()
    query = [
        """
        SELECT files.id, files.path, files.path_key, files.preview_path, files.preview_status,
               files.modified_time, files.size_bytes,
               scores.file_id AS existing_score_id,
               scores.overall_score,
               scores.learned_backend,
               scores.learned_raw_score,
               scores.learned_score_normalized,
               scores.learned_confidence,
               scores.source_modified_time,
               scores.source_size_bytes,
               scores.preset_name,
               scores.model_version,
               scores.image_conversion_version
        FROM files
        LEFT JOIN scores ON scores.file_id = files.id
        """
    ]
    conditions: list[str] = []
    params: list[object] = []

    roots = _resolve_roots_argument(raw_root)
    if roots:
        root_clause, root_params = roots_path_filter("files.path_key", roots)
        conditions.append(root_clause)
        params.extend(root_params)

    if conditions:
        query.append("WHERE " + " AND ".join(conditions))

    query.append("ORDER BY files.id ASC")

    if limit is not None:
        query.append("LIMIT ?")
        params.append(limit)
        if offset:
            query.append("OFFSET ?")
            params.append(offset)
    elif offset:
        query.append("LIMIT -1 OFFSET ?")
        params.append(offset)

    rows = connection.execute(" ".join(query), tuple(params)).fetchall()
    log_duration(
        "scoring.fetch_rows",
        started_at,
        scope="root" if raw_root else "catalog",
        offset=offset,
        limit=limit,
        result_count=len(rows),
    )
    return rows


def count_score_rows(connection, *, raw_root: str | Sequence[str] | Sequence[Path] | None = None) -> int:
    started_at = monotonic_seconds()
    query = [
        """
        SELECT COUNT(files.id) AS row_count
        FROM files
        """
    ]
    params: list[object] = []

    roots = _resolve_roots_argument(raw_root)
    if roots:
        root_clause, root_params = roots_path_filter("files.path_key", roots)
        query.append(f"WHERE {root_clause}")
        params.extend(root_params)

    row = connection.execute(" ".join(query), tuple(params)).fetchone()
    if row is None:
        log_duration("scoring.count_rows", started_at, scope="root" if raw_root else "catalog", total=0)
        return 0
    total = int(row["row_count"] or 0)
    log_duration("scoring.count_rows", started_at, scope="root" if raw_root else "catalog", total=total)
    return total


def delete_score_row(connection, *, file_id: int) -> None:
    connection.execute(
        "DELETE FROM scores WHERE file_id = ?",
        (file_id,),
    )


def needs_score_update(
    row,
    *,
    force: bool,
    learned_backend_name: str | None,
    expected_model_version: str | None = None,
) -> bool:
    if force:
        return True

    # No score row exists yet — always needs scoring.
    if row["existing_score_id"] is None:
        return True

    current_modified_time = row["modified_time"]
    current_size_bytes = row["size_bytes"]
    stored_modified_time = row["source_modified_time"]
    stored_size_bytes = row["source_size_bytes"]

    return (
        row["learned_backend"] != learned_backend_name
        or (expected_model_version is not None and row["model_version"] != expected_model_version)
        or row["learned_raw_score"] is None
        or row["learned_score_normalized"] is None
        or row["overall_score"] is None
        or row["preset_name"] != LEARNED_ONLY_PRESET
        # Treat missing persisted fingerprints on legacy rows as stale until
        # they are rewritten with source metadata from the current file.
        or stored_modified_time is None
        or stored_size_bytes is None
        or _optional_row_value(row, "image_conversion_version") != IMAGE_CONVERSION_VERSION
        or current_modified_time is None
        or current_size_bytes is None
        or stored_modified_time != current_modified_time
        or stored_size_bytes != current_size_bytes
    )


def upsert_score_row(
    connection,
    row,
    *,
    learned: LearnedScoreResult,
    learned_backend: str,
    learned_model_version: str,
) -> None:
    model_version = _db_model_version(learned_model_version)

    connection.execute(
        """
        INSERT INTO scores(
            file_id,
            overall_score,
            learned_backend,
            learned_raw_score,
            learned_score_normalized,
            learned_confidence,
            source_modified_time,
            source_size_bytes,
            preset_name,
            model_version,
            image_conversion_version,
            computed_time
        )
        VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(file_id) DO UPDATE SET
            overall_score = excluded.overall_score,
            learned_backend = excluded.learned_backend,
            learned_raw_score = excluded.learned_raw_score,
            learned_score_normalized = excluded.learned_score_normalized,
            learned_confidence = excluded.learned_confidence,
            source_modified_time = excluded.source_modified_time,
            source_size_bytes = excluded.source_size_bytes,
            preset_name = excluded.preset_name,
            model_version = excluded.model_version,
            image_conversion_version = excluded.image_conversion_version,
            computed_time = excluded.computed_time
        """,
        (
            row["id"],
            learned.normalized_score,
            learned_backend,
            learned.raw_score,
            learned.normalized_score,
            learned.confidence,
            row["modified_time"],
            row["size_bytes"],
            LEARNED_ONLY_PRESET,
            model_version,
            IMAGE_CONVERSION_VERSION,
            utc_now(),
        ),
    )


def select_analysis_path(
    raw_path: str,
    raw_preview_path: str | None,
    preview_status: str | None,
    *,
    preview_conversion_version: str | None = None,
) -> Path | None:
    if (
        raw_preview_path
        and preview_status == "ready"
        and (
            preview_conversion_version is None
            or preview_conversion_version == IMAGE_CONVERSION_VERSION
        )
    ):
        preview_path = Path(raw_preview_path)
        if preview_path.exists():
            return preview_path

    source_path = Path(raw_path)
    # Only accept formats that PIL can open without optional dependencies.
    # .heic/.heif require pillow_heif; if it's missing the model inference
    # path fails explicitly and is treated as unscored. Use the generated
    # preview instead (handled by the preview-generation path above).
    if source_path.exists() and source_path.suffix.casefold() in PIL_ANALYSIS_EXTENSIONS:
        return source_path

    return None
