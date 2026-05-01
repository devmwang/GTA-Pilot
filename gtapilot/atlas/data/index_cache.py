from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

INDEX_CACHE_SCHEMA_VERSION = 4


def _stable_key(payload: dict[str, Any]) -> str:
    return hashlib.sha1(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def load_cached_index(cache_dir: str | Path, payload: dict[str, Any]) -> list[dict[str, Any]] | None:
    cache_dir = Path(cache_dir)
    keyed_payload = {"schema_version": INDEX_CACHE_SCHEMA_VERSION, **payload}
    path = cache_dir / f"{_stable_key(keyed_payload)}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def save_cached_index(
    cache_dir: str | Path,
    payload: dict[str, Any],
    samples: list[dict[str, Any]],
) -> None:
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    keyed_payload = {"schema_version": INDEX_CACHE_SCHEMA_VERSION, **payload}
    path = cache_dir / f"{_stable_key(keyed_payload)}.json"
    path.write_text(json.dumps(samples), encoding="utf-8")
