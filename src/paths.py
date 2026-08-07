"""Path helpers for LANGER 3D structure data."""

from __future__ import annotations

import os
from pathlib import Path

LANGER_ROOT = Path(__file__).resolve().parents[1]


def resolve_data_3d_dir(explicit: str | Path | None = None) -> Path:
    """
    Resolve the 3D structure root directory.

    Priority:
    1. Explicit CLI/path argument (absolute or relative to repo root)
    2. Environment variable ``REE_DATA_3D_ROOT``
    3. ``<repo>/data/data_3d``
    """
    if explicit is not None and str(explicit).strip():
        path = Path(explicit)
        if not path.is_absolute():
            path = LANGER_ROOT / path
        return path.resolve()

    env_path = os.environ.get("REE_DATA_3D_ROOT", "").strip()
    if env_path:
        return Path(env_path).resolve()

    return (LANGER_ROOT / "data" / "data_3d").resolve()
