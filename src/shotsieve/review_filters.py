from __future__ import annotations

from pathlib import Path

from shotsieve.config import RAW_CAMERA_EXTENSIONS
from shotsieve.db import escape_like, roots_path_filter


VALID_DECISION_STATES = {"pending", "delete", "export"}


SORT_ORDERS = {
    "score_asc": "scores.overall_score ASC, files.id ASC",
    "score_desc": "scores.overall_score DESC, scores.file_id ASC",
    "learned_asc": "scores.learned_score_normalized ASC, scores.overall_score ASC, scores.file_id ASC",
    "learned_desc": "scores.learned_score_normalized DESC, files.id ASC",
    "date_asc": "files.capture_time ASC, files.id ASC",
    "date_desc": "files.capture_time DESC, files.id ASC",
    "recent": "files.last_scan_time DESC, files.id DESC",
    "path": "files.path ASC",
    "resolution_asc": "(files.width * files.height) ASC, files.id ASC",
    "resolution_desc": "(files.width * files.height) DESC, files.id ASC",
    "size_asc": "files.size_bytes ASC, files.id ASC",
    "size_desc": "files.size_bytes DESC, files.id ASC",
    "format": "files.format ASC, files.id ASC",
    "width": "files.width ASC, files.id ASC",
    "height": "files.height ASC, files.id ASC",
}


def _build_file_filters(
    *,
    root: str | None = None,
    query: str | None = None,
) -> tuple[list[str], list[object]]:
    path_query = query.casefold() if query else None
    conditions: list[str] = []
    params: list[object] = []

    if root:
        roots = [Path(p).expanduser().resolve() for p in root.split("|") if p.strip()]
        if roots:
            root_clause, root_params = roots_path_filter("files.path_key", roots)
            conditions.append(root_clause)
            params.extend(root_params)

    if path_query:
        conditions.append("unicode_casefold(files.path) LIKE ? ESCAPE '\\'")
        params.append(f"%{escape_like(path_query)}%")

    return conditions, params


def _build_format_filters(*, formats: list[str] | None = None) -> tuple[list[str], list[object]]:
    if not formats:
        return [], []
    
    target_formats: set[str] = set()
    include_other = False
    
    for f in formats:
        f = f.strip().lower()
        if not f:
            continue
        if f == "jpeg":
            target_formats.update({"jpg", "jpeg"})
        elif f == "png":
            target_formats.add("png")
        elif f == "tiff":
            target_formats.update({"tif", "tiff"})
        elif f in ("heif", "heic", "heif/heic"):
            target_formats.update({"heic", "heif", "hif"})
        elif f == "raw":
            target_formats.update({ext.lstrip(".") for ext in RAW_CAMERA_EXTENSIONS})
        elif f == "other":
            include_other = True
        else:
            target_formats.add(f.lstrip("."))
            
    conditions: list[str] = []
    params: list[object] = []
    
    sub_conditions: list[str] = []
    if target_formats:
        placeholders = ", ".join("?" for _ in target_formats)
        sub_conditions.append(f"files.format IN ({placeholders})")
        params.extend(target_formats)
        
    if include_other:
        standard_sets = {"jpg", "jpeg", "png", "tif", "tiff", "heic", "heif", "hif"}
        standard_sets.update({ext.lstrip(".") for ext in RAW_CAMERA_EXTENSIONS})
        placeholders = ", ".join("?" for _ in standard_sets)
        sub_conditions.append(f"files.format NOT IN ({placeholders}) OR files.format IS NULL")
        params.extend(standard_sets)
        
    if sub_conditions:
        conditions.append("(" + " OR ".join(sub_conditions) + ")")
        
    return conditions, params


def _build_resolution_filters(
    *,
    min_mp: float | None = None,
    max_mp: float | None = None,
    min_width: int | None = None,
    max_width: int | None = None,
    min_height: int | None = None,
    max_height: int | None = None,
    min_edge: int | None = None,
    max_edge: int | None = None,
) -> tuple[list[str], list[object]]:
    conditions: list[str] = []
    params: list[object] = []
    
    if min_mp is not None:
        conditions.append("files.width IS NOT NULL AND files.height IS NOT NULL AND (files.width * files.height) >= ?")
        params.append(int(min_mp * 1_000_000))
        
    if max_mp is not None:
        conditions.append("files.width IS NOT NULL AND files.height IS NOT NULL AND (files.width * files.height) <= ?")
        params.append(int(max_mp * 1_000_000))
        
    if min_width is not None:
        conditions.append("files.width IS NOT NULL AND files.width >= ?")
        params.append(min_width)
        
    if max_width is not None:
        conditions.append("files.width IS NOT NULL AND files.width <= ?")
        params.append(max_width)
        
    if min_height is not None:
        conditions.append("files.height IS NOT NULL AND files.height >= ?")
        params.append(min_height)
        
    if max_height is not None:
        conditions.append("files.height IS NOT NULL AND files.height <= ?")
        params.append(max_height)
    
    if min_edge is not None:
        conditions.append("files.width IS NOT NULL AND files.height IS NOT NULL AND (CASE WHEN files.width > files.height THEN files.width ELSE files.height END) >= ?")
        params.append(min_edge)

    if max_edge is not None:
        conditions.append("files.width IS NOT NULL AND files.height IS NOT NULL AND (CASE WHEN files.width > files.height THEN files.width ELSE files.height END) <= ?")
        params.append(max_edge)
        
    return conditions, params


def _build_size_filters(
    *,
    min_size: int | None = None,
    max_size: int | None = None,
) -> tuple[list[str], list[object]]:
    conditions: list[str] = []
    params: list[object] = []
    
    if min_size is not None:
        conditions.append("files.size_bytes IS NOT NULL AND files.size_bytes >= ?")
        params.append(min_size)
        
    if max_size is not None:
        conditions.append("files.size_bytes IS NOT NULL AND files.size_bytes <= ?")
        params.append(max_size)
        
    return conditions, params


def _build_metadata_status_filters(*, metadata: str = "all") -> tuple[list[str], list[object]]:
    conditions: list[str] = []
    
    if metadata not in {"all", "valid", "unknown"}:
        raise ValueError("metadata must be one of: all, valid, unknown")
        
    if metadata == "unknown":
        conditions.append("(files.width IS NULL OR files.height IS NULL OR files.size_bytes IS NULL OR files.format IS NULL)")
    elif metadata == "valid":
        conditions.append("(files.width IS NOT NULL AND files.height IS NOT NULL AND files.size_bytes IS NOT NULL AND files.format IS NOT NULL)")
        
    return conditions, []


def _build_score_filters(
    *,
    require_scored: bool,
    min_score: float | None = None,
    max_score: float | None = None,
) -> tuple[list[str], list[object]]:
    conditions: list[str] = []
    params: list[object] = []

    if require_scored:
        conditions.append("scores.overall_score IS NOT NULL")

    if min_score is not None:
        conditions.append("scores.overall_score >= ?")
        params.append(min_score)

    if max_score is not None:
        conditions.append("scores.overall_score <= ?")
        params.append(max_score)

    return conditions, params


def _build_review_state_filters(*, marked: str = "all") -> tuple[list[str], list[object]]:
    conditions: list[str] = []
    params: list[object] = []

    if marked not in {"all", "delete", "export", "none"}:
        raise ValueError("marked must be one of: all, delete, export, none")

    if marked == "delete":
        conditions.append("COALESCE(review_state.delete_marked, 0) = 1")
    elif marked == "export":
        conditions.append("COALESCE(review_state.export_marked, 0) = 1")
    elif marked == "none":
        conditions.append("COALESCE(review_state.delete_marked, 0) = 0")
        conditions.append("COALESCE(review_state.export_marked, 0) = 0")

    return conditions, params


def _build_issue_filters(*, issues: str = "all") -> tuple[list[str], list[object]]:
    conditions: list[str] = []
    params: list[object] = []

    if issues not in {"all", "issues"}:
        raise ValueError("issues must be one of: all, issues")

    if issues == "issues":
        conditions.append("COALESCE(TRIM(files.last_error), '') <> ''")

    return conditions, params


def _compile_where_clause(*filter_groups: tuple[list[str], list[object]]) -> tuple[str, list[object]]:
    conditions: list[str] = []
    params: list[object] = []
    for group_conditions, group_params in filter_groups:
        conditions.extend(group_conditions)
        params.extend(group_params)

    where_clause = "WHERE " + " AND ".join(conditions) if conditions else ""
    return where_clause, params


def _build_after_id_filter(*, after_id: int | None = None) -> tuple[list[str], list[object]]:
    if after_id is None or after_id <= 0:
        return [], []
    return ["files.id > ?"], [after_id]


def _build_review_browser_where(
    *,
    root: str | None = None,
    marked: str = "all",
    issues: str = "all",
    query: str | None = None,
    min_score: float | None = None,
    max_score: float | None = None,
    formats: list[str] | None = None,
    min_mp: float | None = None,
    max_mp: float | None = None,
    min_width: int | None = None,
    max_width: int | None = None,
    min_height: int | None = None,
    max_height: int | None = None,
    min_edge: int | None = None,
    max_edge: int | None = None,
    min_size: int | None = None,
    max_size: int | None = None,
    metadata: str = "all",
) -> tuple[str, list[object]]:
    """Build the score-backed WHERE clause for the review browser queue."""
    return _compile_where_clause(
        _build_file_filters(root=root, query=query),
        _build_score_filters(require_scored=True, min_score=min_score, max_score=max_score),
        _build_review_state_filters(marked=marked),
        _build_issue_filters(issues=issues),
        _build_format_filters(formats=formats),
        _build_resolution_filters(
            min_mp=min_mp, max_mp=max_mp,
            min_width=min_width, max_width=max_width,
            min_height=min_height, max_height=max_height,
            min_edge=min_edge, max_edge=max_edge,
        ),
        _build_size_filters(min_size=min_size, max_size=max_size),
        _build_metadata_status_filters(metadata=metadata),
    )
