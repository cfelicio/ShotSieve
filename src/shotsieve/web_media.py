from __future__ import annotations

import socket
from dataclasses import dataclass
from http import HTTPStatus
from pathlib import Path
from typing import Any, Callable

MediaPathForFile = Callable[..., Path | None]
DatabaseFactory = Callable[[Path], Any]
BuildConfigFunc = Callable[[str], Any]
PreviewNameFunc = Callable[[Path], str]
PreviewNameCandidatesFunc = Callable[[Path], list[str]]
GuessMediaTypeFunc = Callable[[str], tuple[str | None, str | None]]


@dataclass(frozen=True)
class MediaRequestResult:
    path: Path | None = None
    error_status: HTTPStatus | None = None
    error_message: str | None = None


@dataclass(frozen=True)
class MediaDependencies:
    database: DatabaseFactory
    build_config: BuildConfigFunc
    media_path_for_file: MediaPathForFile
    stable_preview_name: PreviewNameFunc
    preview_name_candidates: PreviewNameCandidatesFunc
    guess_media_type: GuessMediaTypeFunc
    is_within_any_root: Callable[[Path, list[Path]], bool]


def resolve_media_request(
    *,
    db_path: Path,
    file_id: int,
    variant: str,
    dependencies: MediaDependencies,
) -> MediaRequestResult:
    with dependencies.database(db_path) as connection:
        media_path = dependencies.media_path_for_file(connection, file_id=file_id, variant=variant)

    if media_path is None:
        return MediaRequestResult(
            error_status=HTTPStatus.NOT_FOUND,
            error_message="Image not found",
        )

    try:
        resolved_media = media_path.resolve()
        if not resolved_media.is_file():
            return MediaRequestResult(
                error_status=HTTPStatus.NOT_FOUND,
                error_message="Media file does not exist",
            )

        config = dependencies.build_config(str(db_path))
        with dependencies.database(db_path) as root_conn:
            known_roots = [
                Path(str(row["root_path"])).expanduser().resolve()
                for row in root_conn.execute("SELECT DISTINCT root_path FROM scan_runs").fetchall()
                if row["root_path"]
            ]
            if variant == "preview":
                file_row = root_conn.execute(
                    "SELECT path, preview_path FROM files WHERE id = ?",
                    (file_id,),
                ).fetchone()
                if file_row is not None and file_row["path"] and file_row["preview_path"]:
                    preview_candidate = Path(str(file_row["preview_path"])).expanduser().resolve()
                    canonical_preview_name = f"{dependencies.stable_preview_name(Path(str(file_row['path'])))}.jpg"
                    expected_preview_names = {
                        f"{preview_name}.jpg"
                        for preview_name in dependencies.preview_name_candidates(Path(str(file_row["path"])))
                    }
                    preview_reference_count = 0
                    uses_fallback_preview_name = (
                        preview_candidate.name in expected_preview_names
                        and preview_candidate.name != canonical_preview_name
                    )
                    if uses_fallback_preview_name:
                        preview_reference_count = root_conn.execute(
                            "SELECT COUNT(*) AS count FROM files WHERE preview_path = ?",
                            (str(preview_candidate),),
                        ).fetchone()["count"]
                    if (
                        preview_candidate == resolved_media
                        and (
                            preview_candidate.name == canonical_preview_name
                            or (
                                uses_fallback_preview_name
                                and preview_reference_count == 1
                            )
                        )
                    ):
                        known_roots.append(preview_candidate.parent)
        if config.preview_dir:
            known_roots.append(config.preview_dir.resolve())
        if known_roots and not dependencies.is_within_any_root(resolved_media, known_roots):
            return MediaRequestResult(
                error_status=HTTPStatus.FORBIDDEN,
                error_message="Media path is outside allowed roots",
            )
        return MediaRequestResult(path=resolved_media)
    except (OSError, ValueError):
        return MediaRequestResult(
            error_status=HTTPStatus.FORBIDDEN,
            error_message="Invalid media path",
        )


def serve_media_response(
    handler: Any,
    path: Path,
    *,
    guess_media_type: GuessMediaTypeFunc,
    mime_fallbacks: dict[str, str],
) -> None:
    content_type, _ = guess_media_type(path.name)
    file_size = path.stat().st_size
    if not content_type:
        content_type = mime_fallbacks.get(path.suffix.casefold(), "application/octet-stream")

    range_header = handler.headers.get("Range")
    start = 0
    end = file_size - 1

    if range_header and range_header.startswith("bytes="):
        range_spec = range_header[6:].strip()
        if "," not in range_spec:
            parts = range_spec.split("-", 1)
            try:
                if parts[0]:
                    start = int(parts[0])
                    if parts[1]:
                        end = min(int(parts[1]), file_size - 1)
                elif parts[1]:
                    start = max(0, file_size - int(parts[1]))
            except (ValueError, IndexError):
                start = 0
                end = file_size - 1

    if start > end or start >= file_size:
        handler.send_error(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
        return

    content_length = end - start + 1
    if range_header and (start > 0 or end < file_size - 1):
        handler.send_response(HTTPStatus.PARTIAL_CONTENT)
        handler.send_header("Content-Range", f"bytes {start}-{end}/{file_size}")
    else:
        handler.send_response(HTTPStatus.OK)

    handler.send_header("Content-Type", content_type)
    handler.send_header("Content-Length", str(content_length))
    handler.send_header("Accept-Ranges", "bytes")
    handler.send_header("Cache-Control", "max-age=3600, immutable")
    handler.end_headers()
    with path.open("rb") as handle:
        if start > 0:
            handle.seek(start)
        remaining = content_length
        while remaining > 0:
            chunk = handle.read(min(65536, remaining))
            if not chunk:
                break
            try:
                handler.wfile.write(chunk)
            except Exception as exc:
                if isinstance(exc, (BrokenPipeError, ConnectionAbortedError, ConnectionResetError, TimeoutError, socket.timeout)):
                    return
                if isinstance(exc, OSError) and "timed out" in str(exc).casefold():
                    return
                raise
            remaining -= len(chunk)
