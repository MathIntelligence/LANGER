import json
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from src.feature_scaling import GglStandardizer


class BindingAffinityDataset(Dataset):
    def __init__(self, json_file: str):
        self.data = []
        self.ggl_cols: list[str] = []
        self.has_prot = False
        self.prot_dim = 0
        self.ion_dim = 0
        self._ion_scaler_mean: Optional[torch.Tensor] = None
        self._ion_scaler_scale: Optional[torch.Tensor] = None
        self._apply_ion_scaler = False
        self._ggl_standardizer: Optional[GglStandardizer] = None

        with open(json_file, "r", encoding="utf-8") as f:
            try:
                self.data = json.load(f)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Error reading JSON file: {exc}") from exc

        if self.data:
            self.ggl_cols = [col for col in self.data[0].keys() if col.startswith("GGL_")]
            first_item = self.data[0]
            self.has_prot = "prot_embedding" in first_item
            if self.has_prot:
                self.prot_dim = len(first_item["prot_embedding"])
            if "ion_embedding" not in first_item:
                raise ValueError("Embedding JSON must contain 'ion_embedding' in each entry.")
            self.ion_dim = len(first_item["ion_embedding"])

    def __len__(self):
        return len(self.data)

    def has_ggl_features(self) -> bool:
        return len(self.ggl_cols) > 0

    def get_ggl_dim(self) -> int:
        return len(self.ggl_cols)

    def has_protein_embeddings(self) -> bool:
        return self.has_prot

    def get_prot_dim(self) -> int:
        return self.prot_dim

    def get_ion_dim(self) -> int:
        return self.ion_dim

    def set_ion_standardizer(self, mean, scale) -> None:
        self._ion_scaler_mean = torch.tensor(mean, dtype=torch.float32)
        self._ion_scaler_scale = torch.tensor(scale, dtype=torch.float32)
        self._apply_ion_scaler = True

    def set_ggl_standardizer(self, standardizer: GglStandardizer) -> None:
        self._ggl_standardizer = standardizer

    def clear_fold_scalers(self) -> None:
        self._ion_scaler_mean = None
        self._ion_scaler_scale = None
        self._apply_ion_scaler = False
        self._ggl_standardizer = None

    def _scale_ion(self, ion_embedding: torch.Tensor) -> torch.Tensor:
        if self._apply_ion_scaler:
            ion_embedding = (ion_embedding - self._ion_scaler_mean) / self._ion_scaler_scale
        return ion_embedding

    def _scale_ggl(self, ggl_embedding: torch.Tensor) -> torch.Tensor:
        if self._ggl_standardizer is None:
            return ggl_embedding
        scaled = self._ggl_standardizer.transform(ggl_embedding.numpy())
        return torch.tensor(scaled, dtype=torch.float32)

    def __getitem__(self, idx):
        item = self.data[idx]
        if self.has_prot:
            protein_embedding = torch.tensor(item["prot_embedding"], dtype=torch.float32).squeeze(0)
        else:
            protein_embedding = torch.empty(0, dtype=torch.float32)

        ion_embedding = self._scale_ion(
            torch.tensor(item["ion_embedding"], dtype=torch.float32).squeeze(0)
        )
        affinity = torch.tensor(item["affinity"], dtype=torch.float32)

        if self.ggl_cols:
            ggl_raw = torch.tensor(
                [item.get(col, 0.0) for col in self.ggl_cols],
                dtype=torch.float32,
            )
            ggl_embedding = self._scale_ggl(ggl_raw)
            return protein_embedding, ion_embedding, ggl_embedding, affinity

        return protein_embedding, ion_embedding, affinity
