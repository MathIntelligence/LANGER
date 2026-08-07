"""Build LANGER training JSON: ESM-2 protein embeddings + ion features (+ optional GGL)."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from types import SimpleNamespace

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer
from transformers.utils import logging as hf_logging

from src.paths import resolve_data_3d_dir
from src.utils import set_seed

hf_logging.set_verbosity_error()

ESM2_MODEL_NAME = "facebook/esm2_t33_650M_UR50D"
ESM2_MAX_LEN = 1024
GGL_STATS = ("COUNTS", "SUM", "MEAN", "STD", "MIN", "MAX")

DEFAULT_ION_COLUMNS = [
    "EFhands",
    "Atomic Number_metal",
    "Outer shell electrons_metal",
    "First IE_metal  (kJ/mol)",
    "Second IE_metal  (kJ/mol)",
    "Third IE_metal  (kJ/mol)",
    "Electron Affinity_metal  (kJ/mol)",
    "Atomic Radius_metal",
    "Covalent Radius_metal",
    "Pauling EN_metal",
    "Ionic Radius_metal",
]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create LANGER training JSON from data/dataset.csv (ESM-2 + ion ± GGL)."
    )
    parser.add_argument("--input_csv", default="data/dataset.csv", help="Path to input CSV")
    parser.add_argument(
        "--output_json",
        default="data/dataset_embeddings.json",
        help="Path to output JSON",
    )
    parser.add_argument("--sequence_col", default="Sequence", help="Protein sequence column")
    parser.add_argument("--target_col", default="logD", help="Target column")
    parser.add_argument(
        "--ion_label_col",
        default="rare_element",
        help="Ion type label column (used for leave-one-ion-out CV)",
    )
    parser.add_argument(
        "--ion_cols",
        nargs="+",
        default=DEFAULT_ION_COLUMNS,
        help="Ordered ion feature columns used as the ion embedding vector",
    )
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for protein embedding")
    parser.add_argument("--seed", type=int, default=2102, help="Random seed")
    parser.add_argument(
        "--generate_ggl",
        action="store_true",
        help="Generate GGL features from 3D structures and merge into JSON (kernel 0 by default).",
    )
    parser.add_argument(
        "--ggl_data_folder",
        default="data/data_3d",
        help="Folder with per-sample PDB + ion .txt files (extract data/data_3d.tar.gz).",
    )
    parser.add_argument(
        "--ggl_kernel_index",
        type=int,
        default=0,
        help="Kernel index from utils/kernels.csv (default: 0).",
    )
    parser.add_argument(
        "--ggl_cutoff",
        type=float,
        default=12.0,
        help="Distance cutoff for GGL generation.",
    )
    return parser.parse_args()


def preprocess_protein_sequence(sequence: str) -> str:
    return " ".join(re.sub(r"[UZOB]", "X", str(sequence)))


def _normalize_name(name: str) -> str:
    s = str(name).strip().lower()
    s = re.sub(r"\s+", " ", s)
    s = s.replace(" (kj/mol)", "")
    return s


def _normalize_key(v) -> str:
    return str(v).strip()


def _find_col(df: pd.DataFrame, preferred: list[str]) -> str | None:
    name_map = {_normalize_name(c): c for c in df.columns}
    for candidate in preferred:
        norm = _normalize_name(candidate)
        if norm in name_map:
            return name_map[norm]
    return None


def _find_raw_ggl_columns(columns: list[str]) -> list[str]:
    raw_cols = []
    for col in columns:
        if col.startswith("GGL_"):
            raw_cols.append(col)
            continue
        if any(col.endswith(f"_{stat}") for stat in GGL_STATS):
            raw_cols.append(col)
    return raw_cols


def _align_ggl_features_to_input(
    input_df: pd.DataFrame,
    ggl_df: pd.DataFrame,
    input_sequence_col: str,
    input_ion_col: str,
) -> tuple[list[dict], list[str], int]:
    ggl_sequence_col = _find_col(ggl_df, [input_sequence_col, "Sequence", "sequence"])
    ggl_ion_col = _find_col(ggl_df, [input_ion_col, "rare_element", "ion_type", "ion"])
    if ggl_sequence_col is None or ggl_ion_col is None:
        raise ValueError(
            "Could not resolve sequence and ion columns in GGL source. "
            f"Found columns: {list(ggl_df.columns)}"
        )

    raw_ggl_cols = _find_raw_ggl_columns(list(ggl_df.columns))
    if not raw_ggl_cols:
        raise ValueError("No GGL columns found in generated GGL dataframe.")

    ggl_cols = [c if c.startswith("GGL_") else f"GGL_{c}" for c in raw_ggl_cols]

    key_to_features = {}
    for _, row in ggl_df[[ggl_sequence_col, ggl_ion_col] + raw_ggl_cols].iterrows():
        key = (_normalize_key(row[ggl_sequence_col]), _normalize_key(row[ggl_ion_col]))
        key_to_features[key] = {
            gcol: float(row[rcol]) for gcol, rcol in zip(ggl_cols, raw_ggl_cols)
        }

    zero_features = {col: 0.0 for col in ggl_cols}
    aligned = []
    missing = 0
    for _, row in input_df[[input_sequence_col, input_ion_col]].iterrows():
        key = (_normalize_key(row[input_sequence_col]), _normalize_key(row[input_ion_col]))
        feats = key_to_features.get(key)
        if feats is None:
            aligned.append(zero_features.copy())
            missing += 1
        else:
            aligned.append(feats)

    return aligned, ggl_cols, missing


def resolve_ion_columns(df: pd.DataFrame, requested_cols: list[str]) -> list[str]:
    original_cols = list(df.columns)
    normalized_to_original = {_normalize_name(c): c for c in original_cols}

    resolved = []
    missing = []
    for req in requested_cols:
        req_norm = _normalize_name(req)
        if req_norm in normalized_to_original:
            resolved.append(normalized_to_original[req_norm])
            continue

        prefix_matches = [
            orig for norm, orig in normalized_to_original.items() if norm.startswith(req_norm)
        ]
        if len(prefix_matches) == 1:
            resolved.append(prefix_matches[0])
        else:
            missing.append(req)

    if missing:
        raise ValueError(
            f"Missing ion feature columns: {missing}. Available columns: {original_cols}"
        )
    return resolved


def get_protein_embedding(sequences: list[str], model, tokenizer):
    processed_sequences = [preprocess_protein_sequence(seq) for seq in sequences]
    inputs = tokenizer(
        processed_sequences,
        padding=True,
        truncation=True,
        max_length=ESM2_MAX_LEN,
        return_tensors="pt",
        return_attention_mask=True,
    )
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    with torch.no_grad():
        last_hidden_states = model(**inputs).last_hidden_state
        attention_mask = inputs["attention_mask"]
        mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_states.size()).float()
        sum_embeddings = torch.sum(last_hidden_states * mask_expanded, dim=1)
        sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
        embeddings = sum_embeddings / sum_mask

    return embeddings.cpu().numpy()


def generate_ggl_features(
    input_df: pd.DataFrame,
    input_csv: str,
    input_sequence_col: str,
    input_ion_col: str,
    ggl_data_folder: str,
    ggl_kernel_index: int,
    ggl_cutoff: float,
) -> tuple[list[dict], list[str], int]:
    from src.get_ggl_features import GeometricGraphLearningFeatures

    source_csv_path = Path(input_csv)
    if not source_csv_path.is_absolute():
        source_csv_path = _REPO_ROOT / source_csv_path

    data_folder_path = resolve_data_3d_dir(ggl_data_folder)
    ggl_args = SimpleNamespace(
        kernel_index=ggl_kernel_index,
        cutoff=ggl_cutoff,
        path_to_csv=str(source_csv_path),
        data_folder=str(data_folder_path),
        feature_folder=str(_REPO_ROOT / "data" / "ggl_feature_cache"),
        max_rows=None,
    )
    ggl_builder = GeometricGraphLearningFeatures(ggl_args)
    params = {
        "type": ggl_builder.df_kernels.loc[ggl_kernel_index, "type"],
        "power": ggl_builder.df_kernels.loc[ggl_kernel_index, "power"],
        "tau": ggl_builder.df_kernels.loc[ggl_kernel_index, "tau"],
        "cutoff": ggl_cutoff,
    }
    generated_ggl_df = ggl_builder.get_ggl_features(params)
    return _align_ggl_features_to_input(
        input_df=input_df,
        ggl_df=generated_ggl_df,
        input_sequence_col=input_sequence_col,
        input_ion_col=input_ion_col,
    )


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    print(f"Using protein model: esm2 ({ESM2_MODEL_NAME})")
    prot_tokenizer = AutoTokenizer.from_pretrained(ESM2_MODEL_NAME)
    prot_model = AutoModel.from_pretrained(ESM2_MODEL_NAME).to(DEVICE).eval()

    df = pd.read_csv(args.input_csv)
    for col, label in (
        (args.sequence_col, "sequence"),
        (args.target_col, "target"),
        (args.ion_label_col, "ion label"),
    ):
        if col not in df.columns:
            raise ValueError(f"Missing {label} column '{col}'")

    ion_cols = resolve_ion_columns(df, args.ion_cols)
    sequences = df[args.sequence_col].tolist()
    ion_labels = df[args.ion_label_col].astype(str).tolist()
    ion_matrix = df[ion_cols].apply(pd.to_numeric, errors="coerce")
    if ion_matrix.isna().any().any():
        bad_cols = ion_matrix.columns[ion_matrix.isna().any()].tolist()
        raise ValueError(f"Ion feature columns contain non-numeric or missing values: {bad_cols}")
    ion_matrix = ion_matrix.astype(float)

    targets = pd.to_numeric(df[args.target_col], errors="coerce")
    if targets.isna().any():
        raise ValueError(f"Target column '{args.target_col}' contains non-numeric values")

    ggl_rows: list[dict] = []
    ggl_cols: list[str] = []
    if args.generate_ggl:
        ggl_rows, ggl_cols, missing_ggl_rows = generate_ggl_features(
            input_df=df,
            input_csv=args.input_csv,
            input_sequence_col=args.sequence_col,
            input_ion_col=args.ion_label_col,
            ggl_data_folder=args.ggl_data_folder,
            ggl_kernel_index=args.ggl_kernel_index,
            ggl_cutoff=args.ggl_cutoff,
        )
        print(
            "Generated GGL features: "
            f"kernel={args.ggl_kernel_index}, dim={len(ggl_cols)}, missing_rows={missing_ggl_rows}"
        )

    all_embeddings_data = []
    num_entries = len(df)
    num_batches = (num_entries + args.batch_size - 1) // args.batch_size

    for i in tqdm(
        range(0, num_entries, args.batch_size),
        total=num_batches,
        desc="Embedding protein batches",
        unit="batch",
        dynamic_ncols=True,
    ):
        batch_sequences = sequences[i : i + args.batch_size]
        prot_embeds_batch = get_protein_embedding(batch_sequences, prot_model, prot_tokenizer)

        for j in range(len(batch_sequences)):
            row_idx = i + j
            entry = {
                "sequence": sequences[row_idx],
                "prot_embedding": prot_embeds_batch[j].tolist(),
                "ion_embedding": ion_matrix.iloc[row_idx].tolist(),
                "affinity": float(targets.iloc[row_idx]),
                "ion_type": ion_labels[row_idx],
            }
            if ggl_rows:
                entry.update(ggl_rows[row_idx])
            all_embeddings_data.append(entry)

    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(all_embeddings_data, f)

    print(f"Successfully saved embeddings to: {args.output_json}")
    print(f"Total entries processed: {len(all_embeddings_data)}")
    if all_embeddings_data:
        print(f"Protein embedding dim: {len(all_embeddings_data[0]['prot_embedding'])}")
        print(f"Ion embedding dim: {len(all_embeddings_data[0]['ion_embedding'])}")
        if ggl_cols:
            print(f"GGL features dim: {len(ggl_cols)}")
        print("Ion scaler: none (scaled per fold in main.py)")


if __name__ == "__main__":
    main()
