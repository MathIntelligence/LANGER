## LANGER — Lanthanide Affinity Network via Graph and Evolutionary Representations

**LANGER** predicts protein–REE binding affinity (`logD`) from sequence embeddings, ion descriptors, and geometric graph learning (GGL) features.

## Repository layout

| Path | Role |
|------|------|
| `main.py` | Train / evaluate LANGER (ion or cluster CV) |
| `src/dataset_creation.py` | Build embedding JSON (ESM-2 + ion ± GGL) |
| `src/get_ggl_features.py` | GGL feature generation from 3D structures |
| `src/ggl_score.py` | GGL scoring kernels |
| `src/model.py` | LANGER network |
| `data/dataset.csv` | Affinity table + relative PDB paths |
| `data/protein_family_clusters.csv` | Labels for cluster CV |
| `data/cluster_protein_families.py` | Rebuild cluster labels |
| `utils/` | Kernels, ion descriptors, atom/ion radii |

## Model

- **Protein**: ESM-2 650M → 1280-D → `LayerNorm` → `Linear(512)` → ReLU  
- **Ion**: 11 descriptors (raw in JSON; per-fold `StandardScaler` in `main.py`) → `LayerNorm` → `Linear(128)` → ReLU  
- **GGL** (optional, kernel **0**): per-fold `log1p` on `*_COUNTS` + z-score → `LayerNorm` → `Linear(256)` → ReLU  
- **Fusion**: concat → `LayerNorm` → FFN `768 → 384 → 128 → 32 → 1`  
- **Training**: 100 epochs/fold; checkpoint = best val Pearson (cluster) or lowest val MSE (ion)

Expected fold-mean Pearson (protein + ion): **ion ~0.82**, **cluster ~0.57**.

## Setup

```bash
pip install -r requirements.txt
tar -xzf data/data_3d.tar.gz -C data   # only needed for GGL
```

Optional: `export REE_DATA_3D_ROOT=/path/to/data_3d`

## 1) Build embeddings

**Protein + ion:**

```bash
python src/dataset_creation.py \
  --input_csv data/dataset.csv \
  --output_json data/dataset_embeddings.json
```

**Protein + ion + GGL (kernel 0):**

```bash
python src/dataset_creation.py \
  --input_csv data/dataset.csv \
  --output_json data/dataset_embeddings_ggl.json \
  --generate_ggl \
  --ggl_data_folder data/data_3d
```

## 2) Train

**Ion CV:**

```bash
python main.py \
  --dataset_path data/dataset_embeddings.json \
  --split_mode ion \
  --results_csv data/fold_results_ion.csv
```

**Cluster CV:**

```bash
python main.py \
  --dataset_path data/dataset_embeddings.json \
  --split_mode cluster \
  --results_csv data/fold_results_cluster.csv
```

With GGL, point `--dataset_path` at `data/dataset_embeddings_ggl.json`.

The average row in `results_csv` reports fold-mean / pooled `rmse`, `mse`, `pearson`, and `ci`.

## Rebuild clusters (optional)

```bash
python data/cluster_protein_families.py \
  --input_csv data/dataset.csv \
  --output_csv data/protein_family_clusters.csv
```

More detail on tables and paths: [`data/README.md`](data/README.md).
