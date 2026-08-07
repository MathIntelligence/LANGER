# LANGER data

## Files

| Path | Tracked? | Description |
|------|----------|-------------|
| `dataset.csv` | yes | 9,240 rows (616 proteins × 15 REEs): sequence, ion descriptors, `logD`, `af2_jobname`, relative `af2_best_pdb_path` |
| `protein_family_clusters.csv` | yes | Per-sequence cluster labels for `--split_mode cluster` |
| `cluster_protein_families.py` | yes | Rebuild cluster labels (ESM-2 + silhouette sweep) |
| `data_3d.tar.gz` | yes | AF2 PDB + per-ion `.txt` coordinates |
| `data_3d/` | no | Unpacked 3D tree |
| `*.json` | no | Embedding files from `src/dataset_creation.py` |
| `ggl_feature_cache/` | no | Optional GGL cache |

## Extract 3D structures

```bash
tar -xzf data/data_3d.tar.gz -C data
```

Example PDB path in `dataset.csv`:

```text
data/data_3d/seq_0001_038cb041_038cb/seq_0001_038cb041_038cb_best.pdb
```

## Build embeddings

```bash
# Protein + ion
python src/dataset_creation.py \
  --input_csv data/dataset.csv \
  --output_json data/dataset_embeddings.json

# Protein + ion + GGL (kernel 0)
python src/dataset_creation.py \
  --input_csv data/dataset.csv \
  --output_json data/dataset_embeddings_ggl.json \
  --generate_ggl \
  --ggl_data_folder data/data_3d
```

## Rebuild clusters

```bash
python data/cluster_protein_families.py \
  --input_csv data/dataset.csv \
  --output_csv data/protein_family_clusters.csv
```
