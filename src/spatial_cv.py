"""
Spatial cross-validation – block-based spatial splits to ensure
the model generalizes geographically.

Divides the study region into spatial blocks (e.g., 2° × 2°) and
hold out entire blocks for validation/testing.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, log_loss
from sklearn.linear_model import LogisticRegression


def create_spatial_blocks(df, block_size=2.0):
    """
    Assign each sample to a spatial block based on lat/lon quantization.

    Parameters
    ----------
    df : DataFrame with 'latitude' and 'longitude'
    block_size : float – block width in degrees

    Returns
    -------
    df with 'block_id' column
    """
    df = df.copy()
    lat_block = (df["latitude"] // block_size).astype(int)
    lon_block = (df["longitude"] // block_size).astype(int)
    df["block_id"] = lat_block.astype(str) + "_" + lon_block.astype(str)
    n_blocks = df["block_id"].nunique()
    print(f"  Spatial blocks: {n_blocks} blocks of ~{block_size}° × {block_size}°")
    return df


def spatial_kfold(df, feature_col, label_col="ignition", n_folds=5,
                  block_size=2.0):
    """
    Run spatial K-fold cross-validation.

    Each fold holds out a set of spatial blocks. This tests whether
    the model generalizes to unseen geographic regions.

    Parameters
    ----------
    df : DataFrame
    feature_col : str or list – feature column(s)
    label_col : str
    n_folds : int
    block_size : float

    Returns
    -------
    results : dict with mean and std of metrics
    fold_results : list of dicts per fold
    """
    df = create_spatial_blocks(df, block_size)
    blocks = df["block_id"].unique()
    np.random.seed(42)
    np.random.shuffle(blocks)

    fold_size = len(blocks) // n_folds
    fold_results = []

    for fold in range(n_folds):
        test_blocks = blocks[fold * fold_size: (fold + 1) * fold_size]
        test_mask = df["block_id"].isin(test_blocks)
        train_mask = ~test_mask

        train_df = df[train_mask]
        test_df = df[test_mask]

        if isinstance(feature_col, str):
            X_train = train_df[[feature_col]].values
            X_test = test_df[[feature_col]].values
        else:
            X_train = train_df[feature_col].values
            X_test = test_df[feature_col].values

        y_train = train_df[label_col].values.astype(int)
        y_test = test_df[label_col].values.astype(int)

        X_train = np.nan_to_num(X_train, nan=0.0)
        X_test = np.nan_to_num(X_test, nan=0.0)

        if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
            continue

        model = LogisticRegression(
            solver="lbfgs", max_iter=1000, class_weight="balanced"
        )
        model.fit(X_train, y_train)
        y_prob = model.predict_proba(X_test)[:, 1]

        fold_metrics = {
            "fold": fold,
            "n_train": len(train_df),
            "n_test": len(test_df),
            "n_test_blocks": len(test_blocks),
            "auc_roc": roc_auc_score(y_test, y_prob),
            "auc_pr": average_precision_score(y_test, y_prob),
            "log_loss": log_loss(y_test, y_prob),
        }
        fold_results.append(fold_metrics)
        print(f"    Fold {fold}: AUC-ROC={fold_metrics['auc_roc']:.4f}  "
              f"({fold_metrics['n_test_blocks']} blocks held out)")

    fold_df = pd.DataFrame(fold_results)
    summary = {
        "auc_roc_mean": fold_df["auc_roc"].mean(),
        "auc_roc_std": fold_df["auc_roc"].std(),
        "auc_pr_mean": fold_df["auc_pr"].mean(),
        "auc_pr_std": fold_df["auc_pr"].std(),
        "log_loss_mean": fold_df["log_loss"].mean(),
        "log_loss_std": fold_df["log_loss"].std(),
    }
    print(f"\n  Spatial CV summary:")
    print(f"    AUC-ROC: {summary['auc_roc_mean']:.4f} ± {summary['auc_roc_std']:.4f}")
    print(f"    AUC-PR:  {summary['auc_pr_mean']:.4f} ± {summary['auc_pr_std']:.4f}")
    print(f"    LogLoss: {summary['log_loss_mean']:.4f} ± {summary['log_loss_std']:.4f}")

    return summary, fold_df
