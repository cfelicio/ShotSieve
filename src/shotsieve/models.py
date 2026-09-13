from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from typing import Literal


@dataclass(frozen=True, slots=True)
class FilesystemObservation:
    """Result of a single guarded observation of a filesystem path."""

    state: Literal["present", "missing", "unknown"]
    error_text: str | None = None


def observe_filesystem_path(path: Path) -> FilesystemObservation:
    """Observe *path* without turning access errors into ``missing``.

    ``Path.exists()`` intentionally is not used for operation recovery.  On
    some platforms it suppresses permission and other OS errors, which would
    make a transfer look safe to retry when its outcome is actually unknown.
    """
    try:
        path.stat()
    except (FileNotFoundError, NotADirectoryError):
        return FilesystemObservation("missing")
    except OSError as exc:
        return FilesystemObservation("unknown", str(exc) or exc.__class__.__name__)
    return FilesystemObservation("present")


@dataclass(frozen=True, slots=True)
class ScanRunDiagnostic:
    """Durable details for a scan that did not reach its normal commit path."""

    root_path: str
    started_time: str
    completed_time: str
    status: str
    files_seen: int = 0
    files_added: int = 0
    files_updated: int = 0
    files_unchanged: int = 0
    files_removed: int = 0
    error_text: str | None = None


@dataclass(slots=True)
class ScanSummary:
    files_seen: int = 0
    offset_consumed: int = 0
    files_added: int = 0
    files_updated: int = 0
    files_unchanged: int = 0
    files_removed: int = 0
    files_failed: int = 0
    last_batch_error: str | None = None

    def include(self, other: "ScanSummary") -> None:
        self.files_seen += other.files_seen
        self.offset_consumed += other.offset_consumed
        self.files_added += other.files_added
        self.files_updated += other.files_updated
        self.files_unchanged += other.files_unchanged
        self.files_removed += other.files_removed
        self.files_failed += other.files_failed


@dataclass(slots=True)
class FileOperationResult:
    """The durable, per-file result shared by export and delete operations."""

    file_id: int
    source: str
    destination: str | None
    action: str
    outcome: str
    stage: str
    error_text: str | None = None
    errno: int | None = None
    winerror: int | None = None
    retry_safe: bool = False
    source_state: Literal["present", "missing", "unknown"] = "unknown"
    destination_state: Literal["present", "missing", "unknown"] | None = None
    observation_errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "id": self.file_id,
            "file_id": self.file_id,
            "source": self.source,
            "path": self.source,
            "destination": self.destination,
            "action": self.action,
            "requested_action": self.action,
            "outcome": self.outcome,
            "stage": self.stage,
            "error_text": self.error_text,
            "error": self.error_text,
            "errno": self.errno,
            "winerror": self.winerror,
            "retry_safe": self.retry_safe,
            "source_state": self.source_state,
            "destination_state": self.destination_state,
            "observation_errors": list(self.observation_errors),
        }
        return payload


@dataclass(slots=True)
class FileOperationSummary:
    """Aggregate result for a file operation without losing row-level truth."""

    action: str
    items: list[FileOperationResult] = field(default_factory=list)
    warnings: list[dict[str, object]] = field(default_factory=list)
    copied: int = 0
    moved: int = 0
    deleted_ids: list[int] = field(default_factory=list)
    delete_from_disk: bool = False
    cancelled: bool = False
    fatal_error: str | None = None
    contract_enabled: bool = False

    @property
    def completed_count(self) -> int:
        return sum(item.outcome == "success" for item in self.items)

    @property
    def failed_count(self) -> int:
        return sum(item.outcome == "failed" for item in self.items)

    @property
    def partial_count(self) -> int:
        return sum(item.outcome in {"partial", "uncertain", "catalog_failed"} for item in self.items)

    @property
    def unprocessed_count(self) -> int:
        return sum(item.outcome == "unprocessed" for item in self.items)

    @property
    def failed(self) -> list[dict[str, object]]:
        """Legacy failure list, now including partial and unprocessed rows."""
        return [
            item.to_dict()
            for item in self.items
            if item.outcome != "success"
        ]

    @property
    def safe_retry_ids(self) -> list[int]:
        return [item.file_id for item in self.items if item.retry_safe]

    @property
    def outcome(self) -> str:
        if self.cancelled:
            return "cancelled"
        if not self.items:
            return "noop"
        if self.partial_count or self.unprocessed_count:
            return "partial"
        if self.failed_count:
            return "partial" if self.completed_count else "failed"
        return "success"

    def add(self, item: FileOperationResult) -> None:
        self.items.append(item)

    def add_warning(self, *, file_id: int, source: str, stage: str, error: BaseException | str) -> None:
        message = str(error)
        self.warnings.append(
            {
                "file_id": file_id,
                "id": file_id,
                "source": source,
                "stage": stage,
                "error_text": message,
                "error": message,
            }
        )

    def merge(self, other: "FileOperationSummary") -> None:
        self.items.extend(other.items)
        self.warnings.extend(other.warnings)
        self.copied += other.copied
        self.moved += other.moved
        self.deleted_ids.extend(other.deleted_ids)
        self.cancelled = self.cancelled or other.cancelled
        self.contract_enabled = self.contract_enabled or other.contract_enabled
        if other.fatal_error:
            self.fatal_error = other.fatal_error

    def merge_payload(self, payload: dict[str, object]) -> None:
        raw_items = payload.get("items")
        if isinstance(raw_items, list):
            self.contract_enabled = True
            for raw_item in raw_items:
                if not isinstance(raw_item, dict):
                    continue
                try:
                    self.items.append(
                        FileOperationResult(
                            file_id=int(raw_item.get("file_id", raw_item.get("id"))),
                            source=str(raw_item.get("source", raw_item.get("path", ""))),
                            destination=(
                                str(raw_item["destination"])
                                if raw_item.get("destination") is not None
                                else None
                            ),
                            action=str(raw_item.get("action", self.action)),
                            outcome=str(raw_item.get("outcome", "failed")),
                            stage=str(raw_item.get("stage", "unknown")),
                            error_text=(
                                str(raw_item["error_text"])
                                if raw_item.get("error_text") is not None
                                else None
                            ),
                            errno=(int(raw_item["errno"]) if raw_item.get("errno") is not None else None),
                            winerror=(
                                int(raw_item["winerror"])
                                if raw_item.get("winerror") is not None
                                else None
                            ),
                            retry_safe=bool(raw_item.get("retry_safe", False)),
                            source_state=str(raw_item.get("source_state", "unknown")),
                            destination_state=(
                                str(raw_item["destination_state"])
                                if raw_item.get("destination_state") is not None
                                else None
                            ),
                            observation_errors=[
                                str(value)
                                for value in raw_item.get("observation_errors", [])
                            ]
                            if isinstance(raw_item.get("observation_errors", []), list)
                            else [],
                        )
                    )
                except (TypeError, ValueError):
                    continue
        raw_warnings = payload.get("warnings")
        if isinstance(raw_warnings, list):
            self.warnings.extend(item for item in raw_warnings if isinstance(item, dict))
        self.copied += int(payload.get("copied", 0) or 0)
        self.moved += int(payload.get("moved", 0) or 0)
        raw_deleted_ids = payload.get("deleted_ids")
        if isinstance(raw_deleted_ids, list):
            self.deleted_ids.extend(int(file_id) for file_id in raw_deleted_ids)
        self.cancelled = self.cancelled or bool(payload.get("cancelled", False))
        if payload.get("fatal_error"):
            self.fatal_error = str(payload["fatal_error"])

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "action": self.action,
            "copied": self.copied,
            "moved": self.moved,
            "deleted_ids": list(self.deleted_ids),
            "deleted_count": len(self.deleted_ids),
            "delete_from_disk": self.delete_from_disk,
            "failed": self.failed,
            "failed_count": self.failed_count,
            "items": [item.to_dict() for item in self.items],
            "completed_count": self.completed_count,
            "partial_count": self.partial_count,
            "unprocessed_count": self.unprocessed_count,
            "safe_retry_ids": self.safe_retry_ids,
            "warnings": list(self.warnings),
            "cancelled": self.cancelled,
            "fatal_error": self.fatal_error,
            "outcome": self.outcome,
        }
        return payload


def attach_file_operation_summary(
    error: BaseException,
    summary: FileOperationSummary,
    *,
    cancelled: bool = False,
    needs_rollback: bool | None = None,
) -> BaseException:
    """Attach a JSON-ready partial summary without replacing the root cause."""
    summary.cancelled = summary.cancelled or cancelled
    try:
        setattr(error, "file_operation_summary", summary.to_dict())
        setattr(error, "file_operation_cancelled", summary.cancelled)
        if needs_rollback is not None:
            setattr(error, "file_operation_needs_rollback", needs_rollback)
    except Exception:
        # Some third-party exception types can reject custom attributes.  The
        # original exception must still be allowed to propagate truthfully.
        pass
    return error


def operation_summary_from_exception(error: BaseException) -> dict[str, object] | None:
    value: Any = getattr(error, "file_operation_summary", None)
    return value if isinstance(value, dict) else None
