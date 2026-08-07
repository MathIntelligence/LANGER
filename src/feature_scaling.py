"""Per-fold scaling for ion tabular vectors and optional GGL stats."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.preprocessing import StandardScaler


@dataclass
class GglStandardizer:
    mean: np.ndarray
    scale: np.ndarray
    count_indices: list[int]

    def transform(self, values: np.ndarray) -> np.ndarray:
        transformed = np.asarray(values, dtype=np.float64).copy()
        for idx in self.count_indices:
            transformed[idx] = np.log1p(np.maximum(transformed[idx], 0.0))
        return ((transformed - self.mean) / self.scale).astype(np.float32)


def ggl_count_column_indices(column_names: list[str]) -> list[int]:
    return [idx for idx, name in enumerate(column_names) if name.endswith("_COUNTS")]


def fit_ggl_standardizer(train_matrix: np.ndarray, column_names: list[str]) -> GglStandardizer:
    count_indices = ggl_count_column_indices(column_names)
    transformed = np.stack(
        [
            np.asarray(row, dtype=np.float64).copy()
            if not count_indices
            else _log1p_counts(row, count_indices)
            for row in train_matrix
        ],
        axis=0,
    )
    scaler = StandardScaler()
    scaler.fit(transformed)
    return GglStandardizer(
        mean=scaler.mean_.astype(np.float32),
        scale=scaler.scale_.astype(np.float32),
        count_indices=count_indices,
    )


def _log1p_counts(row: np.ndarray, count_indices: list[int]) -> np.ndarray:
    out = np.asarray(row, dtype=np.float64).copy()
    for idx in count_indices:
        out[idx] = np.log1p(np.maximum(out[idx], 0.0))
    return out


def build_ggl_matrix(dataset, indices: list[int], ggl_cols: list[str]) -> np.ndarray:
    rows = [[float(dataset.data[i].get(col, 0.0)) for col in ggl_cols] for i in indices]
    return np.asarray(rows, dtype=np.float32)
