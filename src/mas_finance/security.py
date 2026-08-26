"""Small, dependency-free security helpers used at I/O boundaries."""

from __future__ import annotations

import re
import unicodedata
from pathlib import Path

_SAFE_IDENTIFIER = re.compile(r"[^A-Za-z0-9._-]+")


def safe_identifier(value: str, *, fallback: str = "run", max_length: int = 80) -> str:
    normalized = unicodedata.normalize("NFKC", value).strip()
    normalized = _SAFE_IDENTIFIER.sub("-", normalized).strip(".-_")
    if not normalized:
        normalized = fallback
    return normalized[:max_length]


def safe_upload_name(filename: str, *, allowed_suffixes: frozenset[str] = frozenset({".pdf"})) -> str:
    if "\x00" in filename:
        raise ValueError("filename contains a null byte")
    basename = Path(filename.replace("\\", "/")).name
    if basename in {"", ".", ".."}:
        raise ValueError("invalid upload filename")
    suffix = Path(basename).suffix.lower()
    if suffix not in allowed_suffixes:
        raise ValueError(f"unsupported upload type: {suffix or 'missing suffix'}")
    stem = safe_identifier(Path(basename).stem, fallback="document", max_length=100)
    return f"{stem}{suffix}"


def safe_child(root: Path, filename: str) -> Path:
    resolved_root = root.resolve()
    target = (resolved_root / filename).resolve()
    if target.parent != resolved_root:
        raise ValueError("target path escapes configured root")
    return target
