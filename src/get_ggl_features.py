#!/usr/bin/env python
"""Generate Geometric Graph Learning (GGL) features for protein–REE complexes."""

from __future__ import annotations

import argparse
import sys
import time
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

try:
    from .ggl_score import KernelFunction, SYBYL_GGL
except ImportError:
    from src.ggl_score import KernelFunction, SYBYL_GGL

from src.paths import resolve_data_3d_dir

REPO_ROOT = _REPO_ROOT
PDB_PATH_COLUMNS = ("af2_best_pdb_path", "af2_pdb_path", "best_pdb_path")


def extract_jobname_from_folder(folder_name: str) -> str:
    import re

    match = re.match(r"^(seq_\d+_[0-9a-fA-F]+)(?:_.+)?$", folder_name)
    if match:
        return match.group(1)
    return folder_name


class GeometricGraphLearningFeatures:
    df_kernels = pd.read_csv(REPO_ROOT / "utils" / "kernels.csv")

    def __init__(self, args):
        self.kernel_index = args.kernel_index
        self.cutoff = args.cutoff
        self.path_to_csv = Path(args.path_to_csv).resolve()
        self.data_folder = Path(args.data_folder).resolve()
        self.feature_folder = Path(args.feature_folder).resolve()
        self.max_rows = args.max_rows

    def _build_sample_dir_map(self) -> dict:
        if not self.data_folder.is_dir():
            return {}
        sample_map = {}
        for sample_dir in self.data_folder.iterdir():
            if sample_dir.is_dir():
                sample_map[extract_jobname_from_folder(sample_dir.name)] = sample_dir
        return sample_map

    def _get_protein_file(self, sample_dir: Path) -> Path | None:
        best_pdb_files = sorted(sample_dir.glob("*_best.pdb"))
        if best_pdb_files:
            return best_pdb_files[0]
        pdb_files = sorted(sample_dir.glob("*.pdb"))
        return pdb_files[0] if pdb_files else None

    def _pdb_path_column(self, columns) -> str | None:
        for col in PDB_PATH_COLUMNS:
            if col in columns:
                return col
        return None

    def _resolve_row_structure(self, row, sample_map: dict, pdb_path_col: str | None):
        """Prefer CSV PDB path; fall back to af2_jobname lookup under data_folder."""
        if pdb_path_col is not None and pd.notna(row.get(pdb_path_col)):
            raw_path = str(row[pdb_path_col]).strip()
            if raw_path:
                pdb_path = Path(raw_path)
                if not pdb_path.is_absolute():
                    for candidate in (pdb_path, REPO_ROOT / pdb_path, self.data_folder / pdb_path):
                        if candidate.exists():
                            pdb_path = candidate
                            break
                if pdb_path.exists():
                    return pdb_path, pdb_path.parent

        sample_key = str(row["af2_jobname"]).strip()
        sample_dir = sample_map.get(sample_key)
        if sample_dir is None:
            return None, None

        protein_file = self._get_protein_file(sample_dir)
        if protein_file is None:
            return None, sample_dir
        return protein_file, sample_dir

    def get_ggl_features(self, parameters) -> pd.DataFrame:
        df_data = pd.read_csv(self.path_to_csv)
        if self.max_rows is not None:
            df_data = df_data.head(self.max_rows).copy()

        required_cols = {"af2_jobname", "rare_element"}
        missing_cols = required_cols - set(df_data.columns)
        if missing_cols:
            raise KeyError(f"Input CSV is missing required columns: {sorted(missing_cols)}")

        kernel = KernelFunction(
            kernel_type=parameters["type"],
            kappa=parameters["power"],
            tau=parameters["tau"],
        )
        ggl_model = SYBYL_GGL(Kernel=kernel, cutoff=parameters["cutoff"])
        ggl_stats = ["COUNTS", "SUM", "MEAN", "STD", "MIN", "MAX"]

        sample_map = self._build_sample_dir_map()
        pdb_path_col = self._pdb_path_column(df_data.columns)

        missing_sample_count = 0
        missing_pdb_count = 0
        missing_ion_txt_count = 0
        ggl_features_list = []

        for _, row in df_data.reset_index(drop=True).iterrows():
            ion_name = str(row["rare_element"]).strip()
            protein_file, sample_dir = self._resolve_row_structure(row, sample_map, pdb_path_col)

            if sample_dir is None:
                missing_sample_count += 1
                ggl_features_list.append(np.zeros(37 * 6))
                continue

            if protein_file is None:
                missing_pdb_count += 1
                ggl_features_list.append(np.zeros(37 * 6))
                continue

            ion_txt = sample_dir / f"{ion_name}.txt"
            if not ion_txt.exists():
                missing_ion_txt_count += 1
                ggl_features_list.append(np.zeros(37 * 6))
                continue

            ggl_score = ggl_model.get_ggl_score(
                str(protein_file), str(ion_txt), ion_name=ion_name
            )
            relevant_pairs = [f"{p_atom}-{ion_name}" for p_atom in ggl_model.protein_atom_types]
            ggl_score_filtered = (
                ggl_score.set_index("ATOM_PAIR").reindex(relevant_pairs).fillna(0.0).reset_index()
            )
            ggl_features_list.append(
                ggl_score_filtered.drop(["ATOM_PAIR"], axis=1).values.flatten()
            )

        ggl_feature_columns = [
            f"{atom}_{stat}" for atom, stat in product(ggl_model.protein_atom_types, ggl_stats)
        ]

        df_output = pd.concat(
            [
                df_data.reset_index(drop=True),
                pd.DataFrame(np.array(ggl_features_list), columns=ggl_feature_columns),
            ],
            axis=1,
        )

        print(f"Input rows processed: {len(df_data)}")
        print(
            f"Structure resolution: pdb_path_col={pdb_path_col or 'none'}, "
            f"data_folder={self.data_folder}"
        )
        print(f"Rows with missing sample folder: {missing_sample_count}")
        print(f"Rows with missing protein PDB: {missing_pdb_count}")
        print(f"Rows with missing ion txt file: {missing_ion_txt_count}")
        return df_output

    def main(self) -> None:
        if self.kernel_index is None:
            raise ValueError("--kernel-index is required.")
        parameters = {
            "type": self.df_kernels.loc[self.kernel_index, "type"],
            "power": self.df_kernels.loc[self.kernel_index, "power"],
            "tau": self.df_kernels.loc[self.kernel_index, "tau"],
            "cutoff": self.cutoff,
        }
        df_features = self.get_ggl_features(parameters)
        output_path = (
            self.feature_folder
            / (
                f"{self.path_to_csv.stem}_protein_ree_ker{self.kernel_index}_"
                f"cutoff{self.cutoff}.csv"
            )
        )
        self.feature_folder.mkdir(parents=True, exist_ok=True)
        df_features.to_csv(output_path, index=False, float_format="%.5f")
        print(f"Output file: {output_path}")


def get_args(argv=None):
    default_data_3d = resolve_data_3d_dir(None)
    parser = argparse.ArgumentParser(description="Get GGL features for protein–REE data")
    parser.add_argument(
        "-k", "--kernel-index", type=int, required=True, help="Kernel index (utils/kernels.csv)"
    )
    parser.add_argument("-c", "--cutoff", type=float, default=12.0, help="Binding-site cutoff")
    parser.add_argument(
        "-f",
        "--path_to_csv",
        default=str(REPO_ROOT / "data" / "dataset.csv"),
        help="CSV with af2_jobname, rare_element, and optional af2_best_pdb_path",
    )
    parser.add_argument(
        "-dd",
        "--data_folder",
        type=str,
        default=str(default_data_3d),
        help="Path to data_3d directory",
    )
    parser.add_argument(
        "-fd",
        "--feature_folder",
        type=str,
        default=str(REPO_ROOT / "data" / "ggl_feature_cache"),
        help="Directory for optional CSV feature dumps",
    )
    parser.add_argument("--max-rows", type=int, default=None, help="Optional row limit")
    return parser.parse_args(argv)


def cli_main(argv=None) -> None:
    args = get_args(argv)
    GeometricGraphLearningFeatures(args).main()


if __name__ == "__main__":
    t0 = time.time()
    cli_main()
    print("Done!")
    print("Elapsed time:", time.time() - t0)
