"""
Cluster LANGER proteins into families for cluster-based cross-validation.

Embeds unique sequences with ESM-2, sweeps agglomerative clustering
by silhouette score, and writes cluster labels used by ``main.py --split_mode cluster``.
"""

from __future__ import annotations

import argparse
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer


PROTEIN_MODEL_CONFIGS = {
    "esm2": {
        "model_name": "facebook/esm2_t33_650M_UR50D",
        "max_len": 1024,
        "pooling": "masked_mean",
        "embedding_dim": 1280,
    },
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Estimate protein-family clusters for LANGER using LM embeddings, "
            "agglomerative clustering, and silhouette score."
        )
    )
    parser.add_argument(
        "--input_csv",
        default="data/dataset.csv",
        help="Input CSV (or Excel if the path ends with .xlsx).",
    )
    parser.add_argument("--sheet_name", default=0, help="Excel sheet name/index (xlsx only).")
    parser.add_argument("--sequence_col", default="Sequence", help="Protein sequence column.")
    parser.add_argument(
        "--ion_col",
        default="rare_element",
        help="Ion label column when the input is long-format (one row per sequence–ion).",
    )
    parser.add_argument(
        "--target_col",
        default="logD",
        help="Affinity column pivoted into per-ion columns for long-format inputs.",
    )
    parser.add_argument(
        "--output_csv",
        default="data/protein_family_clusters.csv",
        help="Where to save cluster assignments (one row per unique sequence).",
    )
    parser.add_argument("--batch_size", type=int, default=8, help="Embedding batch size.")
    parser.add_argument("--min_clusters", type=int, default=2, help="Minimum clusters to test.")
    parser.add_argument("--max_clusters", type=int, default=15, help="Maximum clusters to test.")
    parser.add_argument(
        "--fallback_input",
        default=None,
        help="Readable copy if the primary input is locked (OneDrive/Excel).",
    )
    return parser.parse_args()


def preprocess_sequence(sequence: str) -> str:
    return " ".join(
        str(sequence).replace("U", "X").replace("Z", "X").replace("O", "X").replace("B", "X")
    )


def _read_table(path: Path, sheet_name, fallback: str | None = None) -> pd.DataFrame:
    def _load(p: Path) -> pd.DataFrame:
        if p.suffix.lower() in {".xlsx", ".xls"}:
            return pd.read_excel(p, sheet_name=sheet_name)
        return pd.read_csv(p)

    try:
        return _load(path)
    except PermissionError:
        pass

    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_copy = Path(temp_dir) / path.name
            shutil.copy2(path, temp_copy)
            return _load(temp_copy)
    except Exception:
        pass

    if fallback:
        return _load(Path(fallback))

    raise PermissionError(
        f"Cannot read '{path}'. Close it in Excel/OneDrive, or pass --fallback_input."
    )


def to_unique_sequence_table(
    df: pd.DataFrame,
    sequence_col: str,
    ion_col: str,
    target_col: str,
) -> pd.DataFrame:
    """Return one row per sequence; pivot long-format ion affinities when present."""
    if ion_col in df.columns and target_col in df.columns:
        meta_cols = [c for c in ("EFhands", "Efhands") if c in df.columns]
        base = df.drop_duplicates(subset=[sequence_col])[[sequence_col] + meta_cols].copy()
        if "Efhands" in base.columns and "EFhands" not in base.columns:
            base = base.rename(columns={"Efhands": "EFhands"})
        wide = (
            df.pivot_table(
                index=sequence_col,
                columns=ion_col,
                values=target_col,
                aggfunc="first",
            )
            .reset_index()
        )
        wide.columns.name = None
        return base.merge(wide, on=sequence_col, how="left")

    return df.drop_duplicates(subset=[sequence_col]).copy()


def load_protein_encoder(protein_model: str):
    cfg = PROTEIN_MODEL_CONFIGS[protein_model]
    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"])
    model = AutoModel.from_pretrained(cfg["model_name"]).to(DEVICE).eval()
    return tokenizer, model, cfg


def embed_sequences(sequences, tokenizer, model, cfg, batch_size: int) -> np.ndarray:
    embeddings = []
    total_batches = (len(sequences) + batch_size - 1) // batch_size

    for start in tqdm(
        range(0, len(sequences), batch_size),
        total=total_batches,
        desc="Embedding protein sequences",
        unit="batch",
        dynamic_ncols=True,
    ):
        batch_sequences = [preprocess_sequence(seq) for seq in sequences[start : start + batch_size]]
        inputs = tokenizer(
            batch_sequences,
            padding=True,
            truncation=True,
            max_length=cfg["max_len"],
            return_tensors="pt",
            return_attention_mask=True,
        )
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs).last_hidden_state

        if cfg["pooling"] == "mean":
            batch_embeddings = outputs.mean(dim=1)
        else:
            attention_mask = inputs["attention_mask"]
            mask_expanded = attention_mask.unsqueeze(-1).expand(outputs.size()).float()
            summed = torch.sum(outputs * mask_expanded, dim=1)
            denom = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
            batch_embeddings = summed / denom

        embeddings.append(batch_embeddings.cpu().numpy())

    return np.concatenate(embeddings, axis=0)


def choose_n_clusters(embeddings: np.ndarray, min_clusters: int, max_clusters: int):
    n_samples = embeddings.shape[0]
    if n_samples < 3:
        raise ValueError("Need at least 3 sequences to run clustering and silhouette scoring.")

    min_k = max(2, min_clusters)
    max_k = min(max_clusters, n_samples - 1)
    if min_k > max_k:
        raise ValueError(
            f"Invalid cluster range: min_clusters={min_k}, max_clusters={max_k}, n_samples={n_samples}"
        )

    best_k = None
    best_score = -1.0
    score_rows = []

    for k in range(min_k, max_k + 1):
        labels = AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(embeddings)
        if len(np.unique(labels)) < 2:
            continue
        score = silhouette_score(embeddings, labels, metric="euclidean")
        score_rows.append({"n_clusters": k, "silhouette_score": score})
        if score > best_score:
            best_score = score
            best_k = k

    if best_k is None:
        raise RuntimeError("Could not compute a valid silhouette score for any cluster count.")

    return best_k, best_score, pd.DataFrame(score_rows)


def main() -> None:
    args = parse_args()
    input_path = Path(args.input_csv)

    df = _read_table(input_path, args.sheet_name, args.fallback_input)
    if args.sequence_col not in df.columns:
        raise ValueError(
            f"Missing sequence column '{args.sequence_col}'. Available: {list(df.columns)}"
        )

    df = df.dropna(subset=[args.sequence_col]).copy()
    df[args.sequence_col] = df[args.sequence_col].astype(str)

    unique_df = to_unique_sequence_table(
        df, args.sequence_col, args.ion_col, args.target_col
    )
    sequences = unique_df[args.sequence_col].tolist()

    tokenizer, model, cfg = load_protein_encoder("esm2")
    embeddings = embed_sequences(sequences, tokenizer, model, cfg, args.batch_size)

    best_k, best_score, score_df = choose_n_clusters(
        embeddings, args.min_clusters, args.max_clusters
    )
    labels = AgglomerativeClustering(n_clusters=best_k, linkage="ward").fit_predict(embeddings)

    out_df = unique_df.copy()
    out_df["protein_family_cluster"] = labels

    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(output_path, index=False)

    score_csv = output_path.with_name(f"{output_path.stem}_silhouette_scores.csv")
    score_df.to_csv(score_csv, index=False)

    print(f"Estimated number of protein families (best k): {best_k}")
    print(f"Best silhouette score: {best_score:.6f}")
    print(f"Unique sequences clustered: {len(sequences)}")
    print(f"Saved cluster assignments to: {output_path}")
    print(f"Saved silhouette sweep to: {score_csv}")


if __name__ == "__main__":
    main()
