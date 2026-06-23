"""Cache keys for board pipeline runs."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd

from sports_prop_edge.pipeline.board_pipeline import BoardPipelineConfig


def _dataframe_fingerprint(df: pd.DataFrame) -> str:
    if df is None or df.empty:
        return "empty"
    cols = sorted(df.columns.astype(str).tolist())
    blob = df[cols].astype(str).sort_values(cols).to_csv(index=False).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def _history_fingerprint(history_path: Path) -> str:
    if not history_path.exists():
        return "missing"
    stat = history_path.stat()
    return f"{stat.st_mtime_ns}:{stat.st_size}"


def pipeline_cache_key(
    props: pd.DataFrame,
    history_path: Path | str,
    config: BoardPipelineConfig,
) -> str:
    """Stable hash of props content + history file identity + scoring config."""
    hist_path = Path(history_path)
    parts = [
        _dataframe_fingerprint(props),
        _history_fingerprint(hist_path),
        repr(config),
    ]
    return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()
