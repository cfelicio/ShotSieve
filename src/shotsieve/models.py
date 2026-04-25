from __future__ import annotations

from dataclasses import dataclass


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