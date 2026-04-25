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

    parsed = urlparse(f"http://{raw}")
    return parsed.hostname, parsed.port


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

    parsed_origin = urlparse(origin_value)
    if not parsed_origin.scheme or not parsed_origin.hostname:
        return False
    if not is_loopback_host(parsed_origin.hostname):
        return False

    request_host, request_port = host_and_port(host_header)
    origin_port = effective_origin_port(parsed_origin)

    if request_host and not is_loopback_host(request_host):
        return False
    if request_port is None and origin_port is None:
        return True
    if request_port is None or origin_port is None:
        return False
    return request_port == origin_port


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
