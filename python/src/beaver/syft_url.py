"""Helpers for Syft URL parsing and local path translation."""

from __future__ import annotations

import re
import urllib.parse
from pathlib import Path
from typing import Optional


def _load_syftbox_sdk():
    try:
        import syftbox_sdk as syft
    except Exception:
        return None
    return syft


def is_syft_url(value: str) -> bool:
    return isinstance(value, str) and value.startswith("syft://")


def parse_syft_url(url: str) -> tuple[str, str]:
    if not is_syft_url(url):
        raise ValueError(f"Not a syft:// URL: {url}")
    syft = _load_syftbox_sdk()
    if syft is not None:
        parsed = syft.parse_syft_url(url)
        email = getattr(parsed, "email", None)
        path = getattr(parsed, "path", None)
        if email and path is not None:
            return email, path

    remainder = url[len("syft://") :]
    if "/" not in remainder:
        raise ValueError(f"Invalid syft:// URL: {url}")
    email, path = remainder.split("/", 1)
    return email, path


def build_syft_url(email: str, path: str) -> str:
    path = path.lstrip("/")
    syft = _load_syftbox_sdk()
    if syft is not None:
        try:
            return str(syft.build_syft_url(email, path))
        except Exception:
            pass
    return f"syft://{email}/{path}"


def sanitize_syft_path(path_str: str) -> str:
    if not path_str:
        raise ValueError("Syft path cannot be empty")
    if "\x00" in path_str:
        raise ValueError("Syft path contains null byte")

    decoded = urllib.parse.unquote(path_str)
    if "\x00" in decoded:
        raise ValueError("Syft path contains encoded null byte")

    normalized = decoded.replace("\\", "/")
    if normalized.startswith("/"):
        raise ValueError("Syft path must be relative, not absolute")
    if re.match(r"^[A-Za-z]:", normalized):
        raise ValueError("Syft path must be relative, not absolute")
    if normalized.startswith("\\\\"):
        raise ValueError("Syft path must be relative, not UNC")

    parts = []
    for part in normalized.split("/"):
        if not part or part == ".":
            continue
        lower = part.lower()
        if part == "..":
            raise ValueError("Syft path contains traversal")
        if "%2e%2e" in lower or "%2f" in lower:
            raise ValueError("Syft path contains encoded traversal")
        parts.append(part)

    if not parts:
        raise ValueError("Syft path cannot be empty after normalization")
    return "/".join(parts)


def normalize_syft_url(url: str) -> str:
    email, path = parse_syft_url(url)
    path = sanitize_syft_path(path)
    return build_syft_url(email, path)


def syft_url_to_local_path(url: str, *, datasites_root: Path) -> Path:
    email, path = parse_syft_url(url)
    path = sanitize_syft_path(path)
    return Path(datasites_root) / email / Path(*path.split("/"))


def datasites_root_from_path(path: Path) -> Optional[Path]:
    try:
        parts = Path(path).resolve().parts
    except Exception:
        parts = Path(path).parts
    if "datasites" not in parts:
        return None
    idx = parts.index("datasites")
    return Path(*parts[: idx + 1])


def datasites_owner_from_path(path: Path) -> Optional[str]:
    try:
        parts = Path(path).resolve().parts
    except Exception:
        parts = Path(path).parts
    if "datasites" not in parts:
        return None
    idx = parts.index("datasites")
    if len(parts) <= idx + 1:
        return None
    return parts[idx + 1]


def path_to_syft_url(path: Path) -> Optional[str]:
    normalized = str(path).replace("\\", "/")
    parts = [part for part in normalized.split("/") if part]
    if "datasites" not in parts:
        return None
    idx = parts.index("datasites")
    if len(parts) <= idx + 2:
        return None
    email = parts[idx + 1]
    rel = "/".join(parts[idx + 2 :])
    try:
        rel = sanitize_syft_path(rel)
    except ValueError:
        return None
    return build_syft_url(email, rel)
