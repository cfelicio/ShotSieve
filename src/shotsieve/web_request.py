from __future__ import annotations

import json
import select
import socket
import time
from contextlib import contextmanager
from json import JSONDecodeError
from pathlib import Path
from typing import Any, TypedDict


class ScanRequest(TypedDict):
    roots: list[Path]
    preview_dir: str | None
    extensions: str | None
    limit: int | None
    offset: int
    recursive: bool
    rescan_all: bool
    generate_previews: bool
    files_total_hint: int
    resource_profile: str | None


class CompareRequest(TypedDict):
    models: list[str]
    limit: int | None
    offset: int
    root: str | None
    device: str | None
    batch_size: int
    compare_chunk_size: int | None
    resource_profile: str | None


class RequestBodyTimeoutError(TimeoutError):
    """Raised when a request body stalls past the configured socket deadline."""


class DeadlineAwareInput:
    """Wrap a buffered request stream and enforce an absolute read deadline."""

    def __init__(self, raw: Any, *, deadline_getter, connection_getter, poll_timeout_getter) -> None:
        self._raw = raw
        self._deadline_getter = deadline_getter
        self._connection_getter = connection_getter
        self._poll_timeout_getter = poll_timeout_getter
        self._buffer = bytearray()

    def _remaining_deadline_seconds(self, message: str) -> float | None:
        deadline, _ = self._deadline_getter()
        if deadline is None:
            return None
        remaining = float(deadline) - time.monotonic()
        if remaining <= 0:
            raise RequestBodyTimeoutError(message)
        return remaining

    def _configure_socket_timeout(self, remaining: float | None) -> None:
        connection = self._connection_getter()
        if connection is None or remaining is None:
            return
        try:
            connection.settimeout(remaining)
        except OSError:
            pass

    def _read_once(self, size: int, *, message: str) -> bytes:
        while True:
            remaining = self._remaining_deadline_seconds(message)
            connection = self._connection_getter()
            if connection is not None and remaining is not None:
                try:
                    readable, _, _ = select.select([connection], [], [], min(self._poll_timeout_getter(), remaining))
                except OSError:
                    readable = [connection]
                if not readable:
                    continue
            self._configure_socket_timeout(remaining)
            try:
                if hasattr(self._raw, "read1"):
                    chunk = self._raw.read1(size)
                else:
                    chunk = self._raw.read(size)
            except TimeoutError as exc:
                deadline, _ = self._deadline_getter()
                if deadline is None or time.monotonic() < deadline:
                    continue
                raise RequestBodyTimeoutError(message) from exc
            except OSError as exc:
                if not isinstance(exc, socket.timeout) and "timed out" not in str(exc).casefold():
                    raise
                deadline, _ = self._deadline_getter()
                if deadline is None or time.monotonic() < deadline:
                    continue
                raise RequestBodyTimeoutError(message) from exc
            if self._deadline_getter()[0] is not None and time.monotonic() >= self._deadline_getter()[0]:
                raise RequestBodyTimeoutError(message)
            return chunk

    def _consume_buffer(self, size: int) -> bytes:
        take = min(size, len(self._buffer))
        if take <= 0:
            return b""
        chunk = bytes(self._buffer[:take])
        del self._buffer[:take]
        return chunk

    def _read_with_deadline(self, method_name: str, *args, **kwargs):
        method = getattr(self, f"_manual_{method_name}", None)
        if callable(method):
            return method(*args, **kwargs)
        return getattr(self._raw, method_name)(*args, **kwargs)

    def _manual_read(self, size: int = -1) -> bytes:
        _, message = self._deadline_getter()
        read_all = size is None or size < 0
        chunks: list[bytes] = []
        if read_all:
            buffered = self._consume_buffer(len(self._buffer))
            if buffered:
                chunks.append(buffered)
            while True:
                chunk = self._read_once(65_536, message=message)
                if not chunk:
                    break
                chunks.append(chunk)
            return b"".join(chunks)

        remaining = max(0, size)
        buffered = self._consume_buffer(remaining)
        if buffered:
            chunks.append(buffered)
            remaining -= len(buffered)
        while remaining > 0:
            chunk = self._read_once(min(remaining, 65_536), message=message)
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        return b"".join(chunks)

    def _manual_readline(self, size: int = -1) -> bytes:
        _, message = self._deadline_getter()
        limit = 65_537 if size is None or size < 0 else max(1, size)
        chunks: list[bytes] = []
        total = 0
        while total < limit:
            newline_index = self._buffer.find(b"\n")
            if newline_index != -1:
                take = min(newline_index + 1, limit - total)
                chunk = self._consume_buffer(take)
                chunks.append(chunk)
                total += len(chunk)
                break
            if self._buffer:
                take = min(len(self._buffer), limit - total)
                chunk = self._consume_buffer(take)
                chunks.append(chunk)
                total += len(chunk)
                if total >= limit:
                    break
            next_chunk = self._read_once(min(1024, limit - total), message=message)
            if not next_chunk:
                break
            self._buffer.extend(next_chunk)
        return b"".join(chunks)

    def read(self, *args, **kwargs):
        return self._read_with_deadline("read", *args, **kwargs)

    def readline(self, *args, **kwargs):
        return self._read_with_deadline("readline", *args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._raw, name)


def first_value(params: dict[str, list[str]], key: str, default: str | None = None) -> str | None:
    values = params.get(key)
    if not values:
        return default
    value = values[0].strip()
    return value or default


def float_or_none(value: str | None) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except ValueError as exc:
        raise ValueError("Score filters must be numeric") from exc


def try_parse_http_status(value: object) -> int | None:
    try:
        parsed = int(str(value))
    except (TypeError, ValueError):
        return None
    if 100 <= parsed <= 599:
        return parsed
    return None


def required_int(value: object, *, name: str, minimum: int = 0) -> int:
    if value is None or value == "":
        raise ValueError(f"{name} is required")

    try:
        if isinstance(value, bool):
            parsed = int(value)
        elif isinstance(value, int):
            parsed = value
        elif isinstance(value, float):
            parsed = int(value)
        elif isinstance(value, str):
            parsed = int(value)
        else:
            raise TypeError(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer") from exc

    if parsed < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return parsed


def optional_int(value: object, minimum: int = 0) -> int | None:
    if value is None or value == "":
        return None
    return required_int(value, name="value", minimum=minimum)


def int_or_default(
    value: str | None,
    *,
    default: int,
    minimum: int = 0,
    maximum: int | None = None,
) -> int:
    if value is None:
        return default

    parsed = required_int(value, name="value", minimum=minimum)
    if maximum is not None and parsed > maximum:
        raise ValueError(f"value must be at most {maximum}")
    return parsed


def optional_bool(value: object, *, name: str) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    raise ValueError(f"{name} must be a boolean")


def optional_string(value: object) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    raise ValueError("String fields must be strings")


def required_int_list(value: object, *, name: str) -> list[int]:
    if not isinstance(value, list):
        raise ValueError(f"{name} must be a list of integers")
    parsed = [required_int(item, name=name, minimum=1) for item in value]
    if not parsed:
        raise ValueError(f"{name} must not be empty")
    return parsed


def required_choice(value: object, *, name: str, choices: tuple[str, ...]) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} is required")
    normalized = value.strip()
    if normalized not in choices:
        raise ValueError(f"{name} must be one of: {', '.join(choices)}")
    return normalized


def coerce_bool(value: object, *, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    raise ValueError("Boolean fields must be booleans")


def required_path(value: object, *, name: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} is required")
    path = Path(value).expanduser().resolve()
    if not path.exists() or not path.is_dir():
        raise ValueError(f"{name} must point to an existing directory")
    return path


def required_path_list(value: object, *, name: str) -> list[Path]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{name} must be a non-empty list of directories")
    return [required_path(item, name=name) for item in value]


def install_deadline_aware_reader(handler: Any, *, initial_timeout_message: str) -> None:
    raw_reader = getattr(handler, "rfile", None)
    if raw_reader is None or isinstance(raw_reader, DeadlineAwareInput):
        return

    def deadline_getter() -> tuple[float | None, str]:
        deadline = getattr(handler, "_shotsieve_read_deadline_monotonic", None)
        message = getattr(handler, "_shotsieve_read_timeout_message", initial_timeout_message)
        return deadline, message

    def connection_getter() -> Any:
        return getattr(handler, "connection", None)

    def poll_timeout_getter() -> float:
        server = getattr(handler, "server", None)
        return float(getattr(server, "request_io_poll_timeout_seconds", 0.25) or 0.25)

    handler.rfile = DeadlineAwareInput(
        raw_reader,
        deadline_getter=deadline_getter,
        connection_getter=connection_getter,
        poll_timeout_getter=poll_timeout_getter,
    )


def set_handler_read_deadline(handler: Any, *, seconds: float, message: str) -> None:
    handler._shotsieve_read_deadline_monotonic = time.monotonic() + max(0.1, float(seconds))
    handler._shotsieve_read_timeout_message = message


def clear_handler_read_deadline(handler: Any) -> None:
    handler._shotsieve_read_deadline_monotonic = None
    handler._shotsieve_read_timeout_message = "Request read timed out"


@contextmanager
def body_read_deadline(handler: Any):
    connection = getattr(handler, "connection", None)
    previous_timeout = None
    if connection is not None:
        try:
            previous_timeout = connection.gettimeout()
        except OSError:
            previous_timeout = None

    server = getattr(handler, "server", None)
    poll_timeout = float(getattr(server, "request_io_poll_timeout_seconds", 0.25) or 0.25)
    body_timeout = float(getattr(server, "request_body_read_timeout_seconds", 5.0) or 5.0)
    set_handler_read_deadline(handler, seconds=body_timeout, message="Request body read timed out")
    if connection is not None:
        try:
            connection.settimeout(poll_timeout)
        except OSError:
            pass
    try:
        yield
    finally:
        clear_handler_read_deadline(handler)
        if connection is not None:
            try:
                connection.settimeout(previous_timeout)
            except OSError:
                pass


def _read_exact_body(handler: Any, *, content_length: int) -> bytes:
    remaining = content_length
    chunks: list[bytes] = []

    while remaining > 0:
        try:
            chunk = handler.rfile.read(min(65_536, remaining))
        except TimeoutError as exc:
            raise RequestBodyTimeoutError("Request body read timed out") from exc
        except OSError as exc:
            if isinstance(exc, socket.timeout) or "timed out" in str(exc).casefold():
                raise RequestBodyTimeoutError("Request body read timed out") from exc
            raise ValueError("Request body was incomplete") from exc

        if not chunk:
            raise ValueError("Request body was incomplete")

        chunks.append(chunk)
        remaining -= len(chunk)

    return b"".join(chunks)


def read_json_body(handler: Any, *, max_body_size: int) -> dict[str, object]:
    try:
        content_length = int(handler.headers.get("Content-Length", "0"))
    except (TypeError, ValueError) as exc:
        raise ValueError("Content-Length must be an integer") from exc

    if content_length <= 0:
        raise ValueError("Request body is required")
    if content_length > max_body_size:
        raise ValueError("Request body too large")

    with body_read_deadline(handler):
        raw = _read_exact_body(handler, content_length=content_length)
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, JSONDecodeError) as exc:
        raise ValueError("Request body must be valid JSON") from exc

    if not isinstance(payload, dict):
        raise ValueError("Request body must be a JSON object")
    return payload


def parse_scan_request(payload: dict[str, object]) -> ScanRequest:
    return {
        "roots": required_path_list(payload.get("roots"), name="roots"),
        "preview_dir": optional_string(payload.get("preview_dir")),
        "extensions": optional_string(payload.get("extensions")),
        "limit": optional_int(payload.get("limit"), minimum=1),
        "offset": optional_int(payload.get("offset"), minimum=0) or 0,
        "recursive": coerce_bool(payload.get("recursive"), default=True),
        "rescan_all": coerce_bool(payload.get("rescan_all"), default=False),
        "generate_previews": coerce_bool(payload.get("generate_previews"), default=True),
        "files_total_hint": optional_int(payload.get("files_total_hint"), minimum=0) or 0,
        "resource_profile": optional_string(payload.get("resource_profile")),
    }


def parse_compare_request(
    payload: dict[str, object],
    *,
    default_batch_size: int,
) -> CompareRequest:
    model_names = payload.get("models")
    if not isinstance(model_names, list) or not model_names:
        raise ValueError("models must be a non-empty list")
    models = [str(model_name).strip() for model_name in model_names if str(model_name).strip()]
    if not models:
        raise ValueError("models must be a non-empty list")

    return {
        "models": models,
        "limit": optional_int(payload.get("limit"), minimum=1),
        "offset": optional_int(payload.get("offset"), minimum=0) or 0,
        "root": optional_string(payload.get("root")),
        "device": optional_string(payload.get("device")),
        "batch_size": optional_int(payload.get("batch_size"), minimum=1) or default_batch_size,
        "compare_chunk_size": optional_int(payload.get("compare_chunk_size"), minimum=1),
        "resource_profile": optional_string(payload.get("resource_profile")),
    }
