"""Helpers for saving downloaded data without clobbering what is already on disk."""
from __future__ import annotations

from pathlib import Path

import pandas as pd


def merge_replace(old: pd.DataFrame | None, new: pd.DataFrame, key_values, key: str = "ticker") -> pd.DataFrame:
    """Return `old` with every row whose `key` is in `key_values` replaced by the rows of `new`."""
    key_values = set(key_values)
    if old is None or old.empty:
        return new.reset_index(drop=True)
    kept = old[~old[key].isin(key_values)]
    if new.empty:
        return kept.reset_index(drop=True)
    return pd.concat([kept, new], ignore_index=True)


def save_replacing(new: pd.DataFrame, path: Path, key_values, key: str = "ticker") -> pd.DataFrame:
    """Write `new` to a parquet file, replacing only the rows for `key_values` and keeping the rest."""
    path = Path(path)
    old = pd.read_parquet(path) if path.exists() else None
    merged = merge_replace(old, new, key_values, key)
    merged.to_parquet(path, index=False)
    return merged
