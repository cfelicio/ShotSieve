from __future__ import annotations

import ipaddress
from http import HTTPStatus
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlparse


JsonErrorSender = Callable[[HTTPStatus, str], None]


def is_loopback_host(host: str | None) -> bool:
    normalized = (host or "").strip().lower()
    if not normalized:
        return False
    if normalized == "localhost":
        return True

    if normalized.startswith("[") and normalized.endswith("]"):
        normalized = normalized[1:-1]

    if "%" in normalized:
        normalized = normalized.split("%", 1)[0]

    try:
        parsed = ipaddress.ip_address(normalized)
    except ValueError:
        return False

    mapped_ipv4 = getattr(parsed, "ipv4_mapped", None)
    if mapped_ipv4 is not None:
        return bool(mapped_ipv4.is_loopback)
    return bool(parsed.is_loopback)


def is_within_root(candidate: Path, root: Path) -> bool:
    try:
        candidate.relative_to(root)
        return True
    except ValueError:
        return False


def is_within_any_root(candidate: Path, roots: list[Path]) -> bool:
    resolved_candidate = candidate.resolve()
    for root in roots:
        try:
            resolved_root = root.resolve()
        except OSError:
            continue
        if is_within_root(resolved_candidate, resolved_root):
            return True
    return False


def host_and_port(value: str | None) -> tuple[str | None, int | None]:
    raw = (value or "").strip()
    if not raw:
        return None, None

    try:
        parsed = urlparse(f"http://{raw}")
        if (
            not parsed.hostname
            or parsed.username is not None
            or parsed.password is not None
            or parsed.path
            or parsed.params
            or parsed.query
            or parsed.fragment
        ):
            return None, None
        return parsed.hostname, parsed.port
    except ValueError:
        return None, None


def effective_origin_port(parsed_origin: Any) -> int | None:
    if parsed_origin.port is not None:
        return parsed_origin.port
    if parsed_origin.scheme == "http":
        return 80
    if parsed_origin.scheme == "https":
        return 443
    return None


def is_allowed_post_origin(origin: str | None, host_header: str | None) -> bool:
    if origin is None:
        return True

    origin_value = origin.strip()
    if not origin_value or origin_value == "null":
        return False

    try:
        parsed_origin = urlparse(origin_value)
        origin_host = parsed_origin.hostname
        origin_port = effective_origin_port(parsed_origin)
    except ValueError:
        return False
    if parsed_origin.scheme not in {"http", "https"} or not origin_host:
        return False
    if not is_loopback_host(origin_host):
        return False

    request_host, request_port = host_and_port(host_header)
    if request_host is None:
        return False

    if request_host and not is_loopback_host(request_host):
        return False
    if request_port is None and origin_port is None:
        return True
    if request_port is None or origin_port is None:
        return False
    return request_port == origin_port


def is_allowed_local_host(host_header: str | None, *, expected_port: int) -> bool:
    request_host, request_port = host_and_port(host_header)
    if request_host is None or not is_loopback_host(request_host):
        return False
    if request_port is None:
        request_port = 80
    return request_port == expected_port


def reject_non_local_client(
    handler: Any,
    *,
    is_loopback_host_func: Callable[[str | None], bool] = is_loopback_host,
) -> bool:
    client_host = str(handler.client_address[0] if handler.client_address else "").strip()
    if is_loopback_host_func(client_host):
        return False
    handler.send_error(HTTPStatus.FORBIDDEN, "Local access only")
    return True


def reject_disallowed_host(
    handler: Any,
    *,
    send_json_error: JsonErrorSender,
    expected_port: int,
) -> bool:
    if is_allowed_local_host(handler.headers.get("Host"), expected_port=expected_port):
        return False
    send_json_error(HTTPStatus.FORBIDDEN, "Host header is not allowed for local-only requests")
    return True


def reject_disallowed_origin(
    handler: Any,
    *,
    send_json_error: JsonErrorSender,
    is_allowed_post_origin_func: Callable[[str | None, str | None], bool] = is_allowed_post_origin,
) -> bool:
    origin = handler.headers.get("Origin")
    host = handler.headers.get("Host")
    if is_allowed_post_origin_func(origin, host):
        return False
    send_json_error(HTTPStatus.FORBIDDEN, "Origin not allowed for local-only POST requests")
    return True
