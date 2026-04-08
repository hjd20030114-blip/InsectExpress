"""Shared evaluation utilities for the InsectExpress paper figures."""

from pathlib import Path

import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS_DIR = PROJECT_ROOT / "results" / "paper"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

TISSUE_NAMES = [
    "Adult_Brain",
    "Adult_Head",
    "Adult_Midgut",
    "Adult_Hindgut",
    "Adult_FatBody",
    "Adult_MalpighianTubule",
    "Adult_Carcass",
    "Adult_Ovary",
    "Adult_Testis",
    "Larval_Hindgut",
    "Larval_FatBody",
    "Larval_MalpighianTubule",
    "Larval_Midgut",
    "Larval_Carcass",
]

KEY_TISSUES = [
    "Adult_Midgut",
    "Adult_FatBody",
    "Adult_Hindgut",
    "Larval_Midgut",
    "Larval_FatBody",
]
KEY_TISSUE_IDX = [TISSUE_NAMES.index(tissue) for tissue in KEY_TISSUES]

SPECIES_NAMES_V2_7SP = {
    0: "tribolium",
    1: "drosophila",
    2: "silkworm",
    3: "pxyl",
    4: "apis",
    5: "nlug",
    6: "lmig",
}
SPECIES_NAMES_V2_11SP = {
    0: "tribolium",
    1: "drosophila",
    2: "silkworm",
    3: "pxyl",
    4: "nlug",
    5: "lmig",
    6: "agla",
    7: "ldec",
    8: "focc",
    9: "ofur",
    10: "csup",
}
SPECIES_NAMES_V2_12SP = {
    0: "tribolium",
    1: "drosophila",
    2: "silkworm",
    3: "pxyl",
    4: "apis",
    5: "nlug",
    6: "lmig",
    7: "agla",
    8: "ldec",
    9: "focc",
    10: "csup",
    11: "harm",
}


def compute_regression_metrics(y_true, y_pred):
    """Compute regression metrics and return them as a dictionary."""
    if len(y_true) < 3:
        return {
            "pearson": np.nan,
            "spearman": np.nan,
            "r2": np.nan,
            "rmse": np.nan,
            "mae": np.nan,
        }
    pr, _ = pearsonr(y_true, y_pred)
    sr, _ = spearmanr(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    return {"pearson": pr, "spearman": sr, "r2": r2, "rmse": rmse, "mae": mae}


def precision_at_k(y_true, y_pred, k=50):
    """Compute overlap precision between predicted top-k and true top-k genes."""
    if len(y_true) < k:
        k = max(1, len(y_true) // 5)
    true_topk = set(np.argsort(y_true)[-k:])
    pred_topk = set(np.argsort(y_pred)[-k:])
    return len(true_topk & pred_topk) / k


def dcg_at_k(scores, k):
    """Compute DCG@k."""
    scores = np.array(scores[:k])
    return np.sum(scores / np.log2(np.arange(2, len(scores) + 2)))


def ndcg_at_k(y_true, y_pred, k=50):
    """Compute NDCG@k using true expression values as relevance scores."""
    if len(y_true) < k:
        k = max(1, len(y_true) // 5)
    pred_order = np.argsort(y_pred)[::-1][:k]
    ideal_order = np.argsort(y_true)[::-1][:k]
    dcg = dcg_at_k(y_true[pred_order], k)
    idcg = dcg_at_k(y_true[ideal_order], k)
    return dcg / idcg if idcg > 0 else 0.0


def eval_overall(preds, targets, masks):
    """Compute overall metrics on observed entries only."""
    mask_flat = masks.flatten() > 0.5
    y_true = targets.flatten()[mask_flat]
    y_pred = preds.flatten()[mask_flat]
    metrics = compute_regression_metrics(y_true, y_pred)
    metrics["precision_at_50"] = precision_at_k(y_true, y_pred, 50)
    metrics["ndcg_at_50"] = ndcg_at_k(y_true, y_pred, 50)
    metrics["n_samples"] = int(mask_flat.sum())
    return metrics


def eval_per_tissue(preds, targets, masks):
    """Compute per-tissue metrics."""
    results = {}
    for i, tissue in enumerate(TISSUE_NAMES):
        observed = masks[:, i] > 0.5
        if observed.sum() < 3:
            continue
        y_true = targets[observed, i]
        y_pred = preds[observed, i]
        metrics = compute_regression_metrics(y_true, y_pred)
        metrics["precision_at_50"] = precision_at_k(y_true, y_pred, 50)
        metrics["ndcg_at_50"] = ndcg_at_k(y_true, y_pred, 50)
        metrics["n_genes"] = int(observed.sum())
        metrics["is_key_tissue"] = tissue in KEY_TISSUES
        results[tissue] = metrics
    return results


def eval_per_species(preds, targets, masks, species_ids, species_names=None):
    """Compute per-species metrics."""
    if species_names is None:
        species_names = SPECIES_NAMES_V2_12SP
    results = {}
    for species_id in np.unique(species_ids):
        species_mask = species_ids == species_id
        species_name = species_names.get(int(species_id), f"species_{species_id}")
        observed = masks[species_mask].flatten() > 0.5
        if observed.sum() < 3:
            continue
        y_true = targets[species_mask].flatten()[observed]
        y_pred = preds[species_mask].flatten()[observed]
        metrics = compute_regression_metrics(y_true, y_pred)
        metrics["n_genes"] = int(species_mask.sum())
        metrics["n_valid"] = int(observed.sum())
        results[species_name] = metrics
    return results


def eval_key_tissues(preds, targets, masks):
    """Compute pooled metrics across RNAi-relevant tissues."""
    pooled_true = []
    pooled_pred = []
    for tissue_idx in KEY_TISSUE_IDX:
        observed = masks[:, tissue_idx] > 0.5
        pooled_true.append(targets[observed, tissue_idx])
        pooled_pred.append(preds[observed, tissue_idx])
    y_true = np.concatenate(pooled_true)
    y_pred = np.concatenate(pooled_pred)
    metrics = compute_regression_metrics(y_true, y_pred)
    metrics["precision_at_50"] = precision_at_k(y_true, y_pred, 50)
    metrics["ndcg_at_50"] = ndcg_at_k(y_true, y_pred, 50)
    return metrics


def load_npz(path):
    """Load a `val_predictions.npz` file."""
    data = np.load(path, allow_pickle=True)
    return (
        data["preds"],
        data["targets"],
        data["masks"],
        data["species_ids"],
        data["gene_ids"],
    )


def fmt(value, digits=4):
    """Format a scalar for paper tables."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "-"
    return f"{value:.{digits}f}"
