"""Private state vocabulary for filesystem and catalog reconciliation.

The public API deliberately keeps the historical ``FileOperationResult``
shape.  These states are used by operation workers while they decide whether a
row was observed missing, completed, left unprocessed, or made uncertain by a
catalog failure.
"""
from __future__ import annotations

from enum import Enum


class OperationState(str, Enum):
    """Internal row state shared by file-operation workers."""

    COMPLETED = "completed"
    FAILED = "failed"
    OBSERVED_MISSING = "observed_missing"
    DELETED = "deleted"
    UNCERTAIN = "uncertain"
    NOT_PROCESSED = "not_processed"
    CATALOG_UNCERTAIN = "catalog_uncertain"
