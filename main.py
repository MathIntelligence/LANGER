import torch
from src.dataset import BindingAffinityDataset
from src.model import LANGER
import argparse
import copy
import csv
import os
from torch.utils.data import DataLoader, Subset
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from src.utils import set_seed
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.feature_scaling import build_ggl_matrix, fit_ggl_standardizer


def concordance_index(y_true, y_pred):
    n = 0
    h_sum = 0.0
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    for i in range(len(y_true)):
        for j in range(i + 1, len(y_true)):
            if y_true[i] == y_true[j]:
                continue
            n += 1
            diff_pred = y_pred[i] - y_pred[j]
            diff_true = y_true[i] - y_true[j]
            if diff_pred == 0:
                h_sum += 0.5
            elif diff_pred * diff_true > 0:
                h_sum += 1.0

    return h_sum / n if n > 0 else float("nan")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Train LANGER with "
            "leave-one-ion-out or leave-one-cluster-out CV."
        )
    )
    parser.add_argument("--dataset_path", default="data/dataset_embeddings.json", help="Path to embedding dataset JSON.")
    parser.add_argument(
        "--split_mode",
        choices=["ion", "cluster"],
        default="ion",
        help="Data split mode: ion=15-fold by ion type, cluster=5-fold by protein cluster.",
    )

    parser.add_argument("--cluster_csv", default="data/protein_family_clusters.csv", help="CSV file containing sequence-to-cluster assignments.")
    parser.add_argument("--cluster_sequence_col", default="Sequence", help="Sequence column name in cluster_csv.")
    parser.add_argument("--cluster_col", default="protein_family_cluster", help="Cluster label column in cluster_csv.")

    parser.add_argument("--ion_key", default="ion_type", help="JSON key containing ion label in embedding entries.")
    parser.add_argument("--sequence_key", default="sequence", help="JSON key containing sequence in embedding entries.")
    parser.add_argument("--expected_num_ions", type=int, default=15, help="Expected number of unique ions for ion CV.")
    parser.add_argument("--expected_num_clusters", type=int, default=5, help="Expected number of unique protein clusters for cluster CV.")

    parser.add_argument("--device", default="cuda", help="Device preference (cuda or cpu).")
    parser.add_argument("--train_batch_size", type=int, default=256, help="Training batch size.")
    parser.add_argument("--eval_batch_size", type=int, default=256, help="Validation/test batch size.")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--num_epochs", type=int, default=100, help="Number of training epochs per fold.")
    parser.add_argument(
        "--checkpoint_metric",
        choices=["mse", "pearson", "auto"],
        default="auto",
        help=(
            "Checkpoint metric on validation set. "
            "auto=pearson for cluster CV, mse for ion CV."
        ),
    )
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Adam weight decay.")
    parser.add_argument("--scheduler_patience", type=int, default=10, help="ReduceLROnPlateau patience.")
    parser.add_argument("--scheduler_factor", type=float, default=0.2, help="ReduceLROnPlateau factor.")
    parser.add_argument("--val_ratio", type=float, default=0.2, help="Validation split ratio from non-held-out data.")
    parser.add_argument("--prot_input_dim", type=int, default=1280, help="Protein embedding dimension (ESM-2 650M).")
    parser.add_argument("--ion_input_dim", type=int, default=11, help="Ion embedding dimension.")
    parser.add_argument("--split_seed", type=int, default=2102, help="Base seed for train/validation split within each fold.")
    parser.add_argument("--save_dir", default="data/fold_models", help="Directory to save best model per fold.")
    parser.add_argument("--results_csv", default="data/fold_results.csv", help="CSV path for per-fold and averaged results.")
    return parser.parse_args()


def evaluate(model, data_loader, criterion, device, desc=None, has_ggl=False):
    model.eval()
    losses = []
    y_true = []
    y_pred = []

    with torch.no_grad():
        eval_iter = data_loader
        if desc is not None:
            eval_iter = tqdm(
                data_loader,
                desc=desc,
                unit="batch",
                leave=False,
                dynamic_ncols=True,
            )

        for batch in eval_iter:
            if has_ggl:
                proteins, ions, ggl, targets = batch
            else:
                proteins, ions, targets = batch
                ggl = None

            proteins = proteins.to(device)
            ions = ions.to(device)
            targets = targets.to(device).float().view(-1)
            if ggl is not None:
                ggl = ggl.to(device)

            preds = model(proteins, ions, ggl).view(-1)
            loss = criterion(preds, targets)
            losses.append(loss.item())

            y_true.extend(targets.cpu().numpy().tolist())
            y_pred.extend(preds.cpu().numpy().tolist())

    mse = float(np.mean((np.array(y_true) - np.array(y_pred)) ** 2))
    rmse = float(np.sqrt(mse))
    try:
        pearson = float(pearsonr(y_true, y_pred)[0])
    except Exception:
        pearson = float("nan")
    ci = float(concordance_index(y_true, y_pred))
    mean_loss = float(np.mean(losses)) if losses else float("nan")

    return {
        "loss": mean_loss,
        "mse": mse,
        "rmse": rmse,
        "pearson": pearson,
        "ci": ci,
        "y_true": y_true,
        "y_pred": y_pred,
    }


def _sanitize_label(label: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in label)


def _split_train_val_indices(trainval_indices, ion_labels, val_ratio, fold_seed):
    val_size = max(1, int(val_ratio * len(trainval_indices)))
    if val_size >= len(trainval_indices):
        val_size = len(trainval_indices) - 1

    trainval_positions = list(range(len(trainval_indices)))
    trainval_ion_labels = [ion_labels[idx] for idx in trainval_indices]
    val_size = min(val_size, len(trainval_positions) - 1)
    try:
        train_pos, val_pos = train_test_split(
            trainval_positions,
            test_size=val_size,
            random_state=fold_seed,
            stratify=trainval_ion_labels,
        )
    except ValueError:
        train_pos, val_pos = train_test_split(
            trainval_positions,
            test_size=val_size,
            random_state=fold_seed,
            stratify=None,
        )
    train_indices = [trainval_indices[p] for p in train_pos]
    val_indices = [trainval_indices[p] for p in val_pos]
    return train_indices, val_indices


def resolve_checkpoint_metric(args) -> str:
    if args.checkpoint_metric != "auto":
        return args.checkpoint_metric
    return "pearson" if args.split_mode == "cluster" else "mse"


def _ions_already_globally_scaled(dataset, max_samples: int = 512) -> bool:
    """Detect JSON built with --ion_scaler standard/minmax (avoid double scaling)."""
    n = min(len(dataset), max_samples)
    if n == 0:
        return False
    values = np.asarray([dataset.data[i]["ion_embedding"] for i in range(n)], dtype=np.float64)
    dim_std = values.std(axis=0)
    dim_mean = np.abs(values.mean(axis=0))
    return bool(
        np.all(dim_std > 0.2)
        and np.all(dim_std < 2.5)
        and np.all(dim_mean < 0.75)
        and np.max(np.abs(values)) < 8.0
    )


def _fit_fold_scalers(dataset, train_indices, scale_ggl: bool) -> None:
    dataset.clear_fold_scalers()
    if _ions_already_globally_scaled(dataset):
        print("Ion vectors look globally scaled in JSON; skipping per-fold ion StandardScaler.")
    else:
        ion_train = np.asarray(
            [dataset.data[i]["ion_embedding"] for i in train_indices],
            dtype=np.float32,
        )
        ion_scaler = StandardScaler()
        ion_scaler.fit(ion_train)
        dataset.set_ion_standardizer(ion_scaler.mean_, ion_scaler.scale_)

    if scale_ggl and dataset.has_ggl_features():
        ggl_train = build_ggl_matrix(dataset, train_indices, dataset.ggl_cols)
        dataset.set_ggl_standardizer(fit_ggl_standardizer(ggl_train, dataset.ggl_cols))


def train_one_fold(args, dataset, fold_tag, trainval_indices, test_indices, fold_idx, device, ion_labels):
    fold_seed = args.split_seed + fold_idx
    set_seed(fold_seed)

    if len(trainval_indices) < 2:
        raise ValueError(f"Fold '{fold_tag}' has insufficient non-test samples for train/val split.")
    if len(test_indices) < 1:
        raise ValueError(f"Fold '{fold_tag}' has no test samples.")

    train_indices, val_indices = _split_train_val_indices(
        trainval_indices,
        ion_labels,
        args.val_ratio,
        fold_seed,
    )
    train_data = Subset(dataset, train_indices)
    val_data = Subset(dataset, val_indices)
    test_data = Subset(dataset, test_indices)

    _fit_fold_scalers(dataset, train_indices, scale_ggl=dataset.has_ggl_features())

    train_loader = DataLoader(train_data, batch_size=args.train_batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=args.eval_batch_size, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=args.eval_batch_size, shuffle=False)

    has_ggl = dataset.has_ggl_features()
    ggl_dim = dataset.get_ggl_dim() if has_ggl else 0

    model = LANGER(
        prot_input_dim=args.prot_input_dim,
        ion_input_dim=args.ion_input_dim,
        ggl_input_dim=ggl_dim,
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    criterion = nn.MSELoss()
    scheduler = ReduceLROnPlateau(
        optimizer,
        factor=args.scheduler_factor,
        patience=args.scheduler_patience,
    )

    checkpoint_metric = resolve_checkpoint_metric(args)
    best_val_mse = float("inf")
    best_val_pearson = float("-inf")
    best_state = None

    epoch_iter = tqdm(
        range(args.num_epochs),
        desc=f"Fold {fold_idx+1} ({fold_tag}) [{checkpoint_metric}]",
        unit="epoch",
        leave=True,
        dynamic_ncols=True,
    )

    for epoch in epoch_iter:
        model.train()
        train_iter = tqdm(
            train_loader,
            desc=f"Fold {fold_idx+1} train {epoch+1}/{args.num_epochs}",
            unit="batch",
            leave=False,
            dynamic_ncols=True,
        )
        running_loss = 0.0
        batch_count = 0
        for batch in train_iter:
            if has_ggl:
                proteins, ions, ggl, targets = batch
            else:
                proteins, ions, targets = batch
                ggl = None

            proteins = proteins.to(device)
            ions = ions.to(device)
            targets = targets.to(device).float().view(-1)
            if ggl is not None:
                ggl = ggl.to(device)

            optimizer.zero_grad()
            preds = model(proteins, ions, ggl).view(-1)
            loss = criterion(preds, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            batch_count += 1
            train_iter.set_postfix(train_mse_loss=running_loss / batch_count)

        val_metrics = evaluate(
            model,
            val_loader,
            criterion,
            device,
            desc=f"Fold {fold_idx+1} validate {epoch+1}/{args.num_epochs}",
            has_ggl=has_ggl,
        )
        scheduler.step(val_metrics["mse"])

        if checkpoint_metric == "pearson":
            val_score = val_metrics["pearson"]
            if not np.isnan(val_score):
                better_pearson = val_score > (best_val_pearson + 1e-12)
                tied_pearson = abs(val_score - best_val_pearson) <= 1e-12
                if better_pearson or (tied_pearson and val_metrics["mse"] < best_val_mse):
                    best_val_pearson = val_score
                    best_val_mse = val_metrics["mse"]
                    best_state = copy.deepcopy(model.state_dict())
        elif val_metrics["mse"] < best_val_mse:
            best_val_mse = val_metrics["mse"]
            best_val_pearson = val_metrics["pearson"]
            best_state = copy.deepcopy(model.state_dict())

        epoch_iter.set_postfix(
            best_val_mse=f"{best_val_mse:.6f}",
            val_mse=f"{val_metrics['mse']:.6f}",
            val_pearson=f"{val_metrics['pearson']:.4f}",
        )

    if best_state is None:
        raise RuntimeError(f"No best model state captured for fold '{fold_tag}'")

    model.load_state_dict(best_state)
    os.makedirs(args.save_dir, exist_ok=True)
    model_path = os.path.join(
        args.save_dir,
        f"best_model_fold_{fold_idx+1}_{_sanitize_label(str(fold_tag))}.pth",
    )
    torch.save(model.state_dict(), model_path)

    test_metrics = evaluate(
        model,
        test_loader,
        criterion,
        device,
        desc=f"Fold {fold_idx+1} test ({fold_tag})",
        has_ggl=has_ggl,
    )

    dataset.clear_fold_scalers()

    return best_val_mse, test_metrics, model_path


def build_sequence_to_cluster_map(args):
    cluster_df = pd.read_csv(args.cluster_csv)
    for col in [args.cluster_sequence_col, args.cluster_col]:
        if col not in cluster_df.columns:
            raise ValueError(
                f"Missing '{col}' in cluster CSV. Available columns: {list(cluster_df.columns)}"
            )

    cluster_df = cluster_df[[args.cluster_sequence_col, args.cluster_col]].dropna()
    cluster_df[args.cluster_sequence_col] = cluster_df[args.cluster_sequence_col].astype(str)

    return (
        cluster_df.drop_duplicates(subset=[args.cluster_sequence_col])
        .set_index(args.cluster_sequence_col)[args.cluster_col]
        .to_dict()
    )


def build_fold_definitions(args, dataset):
    ion_labels = []
    sequences = []
    for item in dataset.data:
        if args.ion_key not in item:
            raise ValueError(
                f"Missing '{args.ion_key}' in embedding JSON entries. "
                "Regenerate embeddings with src/dataset_creation.py using --ion_label_col rare_element."
            )
        if args.sequence_key not in item:
            raise ValueError(
                f"Missing '{args.sequence_key}' in embedding JSON entries. "
                "Regenerate embeddings with src/dataset_creation.py to include sequence metadata."
            )
        ion_labels.append(str(item[args.ion_key]))
        sequences.append(str(item[args.sequence_key]))

    folds = []

    if args.split_mode == "ion":
        unique_ions = sorted(set(ion_labels))
        if len(unique_ions) != args.expected_num_ions:
            raise ValueError(
                f"Expected {args.expected_num_ions} unique ions, found {len(unique_ions)}: {unique_ions}"
            )

        for i, held_out_ion in enumerate(unique_ions):
            test_indices = [idx for idx, ion in enumerate(ion_labels) if ion == held_out_ion]
            trainval_indices = [idx for idx, ion in enumerate(ion_labels) if ion != held_out_ion]
            folds.append(
                {
                    "fold_idx": i,
                    "fold_tag": f"ion={held_out_ion}",
                    "held_out_ion": held_out_ion,
                    "held_out_cluster": "",
                    "trainval_indices": trainval_indices,
                    "test_indices": test_indices,
                }
            )
        return folds

    sequence_to_cluster = build_sequence_to_cluster_map(args)
    sample_clusters = []
    for seq in sequences:
        if seq not in sequence_to_cluster:
            raise ValueError(
                "Found sequence in embedding dataset that is not present in cluster CSV. "
                "Make sure cluster CSV was generated from the same sequence set."
            )
        sample_clusters.append(sequence_to_cluster[seq])

    if args.split_mode == "cluster":
        unique_clusters = sorted(set(sample_clusters))
        if len(unique_clusters) != args.expected_num_clusters:
            raise ValueError(
                f"Expected {args.expected_num_clusters} unique clusters, found {len(unique_clusters)}: {unique_clusters}"
            )

        for i, held_out_cluster in enumerate(unique_clusters):
            test_indices = [idx for idx, cluster in enumerate(sample_clusters) if cluster == held_out_cluster]
            trainval_indices = [idx for idx, cluster in enumerate(sample_clusters) if cluster != held_out_cluster]
            folds.append(
                {
                    "fold_idx": i,
                    "fold_tag": f"cluster={held_out_cluster}",
                    "held_out_ion": "",
                    "held_out_cluster": held_out_cluster,
                    "trainval_indices": trainval_indices,
                    "test_indices": test_indices,
                }
            )
        return folds

    raise ValueError(f"Unsupported split mode: {args.split_mode}")


def write_results_csv(results, output_csv, global_metrics=None):
    os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
    fieldnames = [
        "split_mode",
        "fold",
        "held_out_ion",
        "held_out_cluster",
        "best_val_mse",
        "rmse_mean",
        "rmse_global",
        "mse_mean",
        "mse_global",
        "pearson_mean",
        "pearson_global",
        "ci_mean",
        "ci_global",
        "model_path",
    ]

    avg_row = {
        "split_mode": "average",
        "fold": "average",
        "held_out_ion": "average",
        "held_out_cluster": "average",
        "best_val_mse": float(np.nanmean([r["best_val_mse"] for r in results])),
        "rmse_mean": float(np.nanmean([r["rmse"] for r in results])),
        "mse_mean": float(np.nanmean([r["mse"] for r in results])),
        "pearson_mean": float(np.nanmean([r["pearson"] for r in results])),
        "ci_mean": float(np.nanmean([r["ci"] for r in results])),
        "model_path": "",
    }

    if global_metrics:
        avg_row["rmse_global"] = float(global_metrics["rmse"])
        avg_row["mse_global"] = float(global_metrics["mse"])
        avg_row["pearson_global"] = float(global_metrics["pearson"])
        avg_row["ci_global"] = float(global_metrics["ci"])
    else:
        avg_row["rmse_global"] = float("nan")
        avg_row["mse_global"] = float("nan")
        avg_row["pearson_global"] = float("nan")
        avg_row["ci_global"] = float("nan")

    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            row["rmse_global"] = float("nan")
            row["mse_global"] = float("nan")
            row["pearson_global"] = float("nan")
            row["ci_global"] = float("nan")
            row["rmse_mean"] = row.pop("rmse")
            row["mse_mean"] = row.pop("mse")
            row["pearson_mean"] = row.pop("pearson")
            row["ci_mean"] = row.pop("ci")
            writer.writerow(row)
        writer.writerow(avg_row)


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    dataset = BindingAffinityDataset(args.dataset_path)
    if dataset.has_protein_embeddings():
        detected_prot_dim = dataset.get_prot_dim()
        if args.prot_input_dim <= 0:
            args.prot_input_dim = detected_prot_dim
        elif args.prot_input_dim != detected_prot_dim:
            print(
                "Warning: --prot_input_dim does not match JSON prot embedding dim "
                f"({args.prot_input_dim} vs {detected_prot_dim}). Using detected dim."
            )
            args.prot_input_dim = detected_prot_dim
    else:
        args.prot_input_dim = 0
        print("Dataset has no protein embeddings; training ion-only backbone (+ optional GGL).")

    detected_ion_dim = dataset.get_ion_dim()
    if args.ion_input_dim != detected_ion_dim:
        print(
            "Warning: --ion_input_dim does not match JSON ion embedding dim "
            f"({args.ion_input_dim} vs {detected_ion_dim}). Using detected dim."
        )
        args.ion_input_dim = detected_ion_dim

    print(f"Total dataset size: {len(dataset)}")
    print(
        f"Checkpoint metric: {resolve_checkpoint_metric(args)} "
        f"(split_mode={args.split_mode})"
    )

    ion_labels = []
    for item in dataset.data:
        if args.ion_key not in item:
            raise ValueError(f"Missing '{args.ion_key}' in embedding JSON entries.")
        ion_labels.append(str(item[args.ion_key]))

    folds = build_fold_definitions(args, dataset)

    all_results = []
    all_y_true = []
    all_y_pred = []

    fold_iter = tqdm(folds, desc=f"{args.split_mode} folds", unit="fold", dynamic_ncols=True)
    for fold_cfg in fold_iter:
        best_val_mse, test_metrics, model_path = train_one_fold(
            args,
            dataset,
            fold_cfg["fold_tag"],
            fold_cfg["trainval_indices"],
            fold_cfg["test_indices"],
            fold_cfg["fold_idx"],
            device,
            ion_labels,
        )
        fold_result = {
            "split_mode": args.split_mode,
            "fold": fold_cfg["fold_idx"] + 1,
            "held_out_ion": fold_cfg["held_out_ion"],
            "held_out_cluster": fold_cfg["held_out_cluster"],
            "best_val_mse": best_val_mse,
            "rmse": test_metrics["rmse"],
            "mse": test_metrics["mse"],
            "pearson": test_metrics["pearson"],
            "ci": test_metrics["ci"],
            "model_path": model_path,
        }
        all_results.append(fold_result)

        if "y_true" in test_metrics and "y_pred" in test_metrics:
            all_y_true.extend(test_metrics["y_true"])
            all_y_pred.extend(test_metrics["y_pred"])
        fold_iter.set_postfix(
            last_fold=fold_result["fold"],
            ion=str(fold_result["held_out_ion"]),
            cluster=str(fold_result["held_out_cluster"]),
            last_val_mse=f"{fold_result['best_val_mse']:.6f}",
        )
        print(
            f"Fold {fold_result['fold']} done | split={args.split_mode}, "
            f"held_out_ion={fold_result['held_out_ion']}, "
            f"held_out_cluster={fold_result['held_out_cluster']} | "
            f"best_val_mse={fold_result['best_val_mse']:.6f}, "
            f"rmse={fold_result['rmse']:.6f}, mse={fold_result['mse']:.6f}, "
            f"pearson={fold_result['pearson']:.6f}, ci={fold_result['ci']:.6f}"
        )

    global_metrics = None
    if all_y_true and all_y_pred:
        all_y_true_arr = np.array(all_y_true)
        all_y_pred_arr = np.array(all_y_pred)
        global_mse = float(np.mean((all_y_true_arr - all_y_pred_arr) ** 2))
        global_rmse = float(np.sqrt(global_mse))
        try:
            global_pearson = float(pearsonr(all_y_true_arr, all_y_pred_arr)[0])
        except Exception:
            global_pearson = float("nan")
        global_ci = float(concordance_index(all_y_true_arr, all_y_pred_arr))
        global_metrics = {
            "rmse": global_rmse,
            "mse": global_mse,
            "pearson": global_pearson,
            "ci": global_ci,
        }

    write_results_csv(all_results, args.results_csv, global_metrics)
    print(f"Saved fold results to: {args.results_csv}")


if __name__ == '__main__':
    main()
