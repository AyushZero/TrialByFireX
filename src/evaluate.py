"""
Evaluation metrics and diagnostic plots.

Metrics
───────
  • AUC-ROC  – discrimination under class imbalance
  • AUC-PR   – performance on rare positive events
  • Brier score – calibration quality
  • Reliability diagram
  • Hit rate vs false-alarm rate by strata (slope, NDVI)
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_auc_score, average_precision_score, brier_score_loss,
    roc_curve, precision_recall_curve, confusion_matrix,
    classification_report,
)


# ══════════════════════════════════════════════════════════════════
# Scalar metrics
# ══════════════════════════════════════════════════════════════════

def compute_csi(y_true, y_prob, threshold=0.5):
    """
    Critical Success Index (CSI) / Threat Score.

    CSI = Hits / (Hits + Misses + False Alarms)

    Commonly used in weather/fire forecasting to assess rare event detection.
    Range: [0, 1], higher is better. Ignores correct negatives.

    Parameters
    ----------
    y_true : array-like - ground truth binary labels
    y_prob : array-like - predicted probabilities
    threshold : float - decision threshold

    Returns
    -------
    float - CSI score
    """
    y_pred = (np.asarray(y_prob) >= threshold).astype(int)
    y_true = np.asarray(y_true)

    hits = ((y_pred == 1) & (y_true == 1)).sum()
    misses = ((y_pred == 0) & (y_true == 1)).sum()
    false_alarms = ((y_pred == 1) & (y_true == 0)).sum()

    denominator = hits + misses + false_alarms
    if denominator == 0:
        return 0.0
    return hits / denominator


def compute_gilbert_skill_score(y_true, y_prob, threshold=0.5):
    """
    Gilbert Skill Score (GSS) / Equitable Threat Score (ETS).

    Adjusts CSI for hits expected by random chance:
    GSS = (Hits - Hits_random) / (Hits + Misses + FA - Hits_random)

    Range: [-1/3, 1], where 0 = no skill, 1 = perfect.
    More robust than CSI for rare events.

    Parameters
    ----------
    y_true : array-like - ground truth binary labels
    y_prob : array-like - predicted probabilities
    threshold : float - decision threshold

    Returns
    -------
    float - Gilbert Skill Score
    """
    y_pred = (np.asarray(y_prob) >= threshold).astype(int)
    y_true = np.asarray(y_true)

    hits = ((y_pred == 1) & (y_true == 1)).sum()
    misses = ((y_pred == 0) & (y_true == 1)).sum()
    false_alarms = ((y_pred == 1) & (y_true == 0)).sum()
    correct_neg = ((y_pred == 0) & (y_true == 0)).sum()

    total = hits + misses + false_alarms + correct_neg
    if total == 0:
        return 0.0

    # Expected hits by random chance
    hits_random = ((hits + misses) * (hits + false_alarms)) / total

    denominator = hits + misses + false_alarms - hits_random
    if denominator == 0:
        return 0.0
    return (hits - hits_random) / denominator


def compute_metrics(y_true, y_prob, threshold=0.5):
    """
    Compute core evaluation metrics.

    Returns dict with: auc_roc, auc_pr, brier_score, csi, gss
    """
    metrics = {}
    try:
        metrics["auc_roc"]     = roc_auc_score(y_true, y_prob)
    except ValueError:
        metrics["auc_roc"]     = float("nan")
    try:
        metrics["auc_pr"]      = average_precision_score(y_true, y_prob)
    except ValueError:
        metrics["auc_pr"]      = float("nan")
    metrics["brier_score"] = brier_score_loss(y_true, y_prob)
    metrics["csi"]         = compute_csi(y_true, y_prob, threshold)
    metrics["gss"]         = compute_gilbert_skill_score(y_true, y_prob, threshold)
    return metrics


def print_metrics(name, metrics):
    """Pretty-print a metrics dict."""
    print(f"\n  {'─'*40}")
    print(f"  Model: {name}")
    for k, v in metrics.items():
        print(f"    {k:>15s}: {v:.4f}")


# ══════════════════════════════════════════════════════════════════
# Comparison table
# ══════════════════════════════════════════════════════════════════

def compare_models(results_dict, save_path=None):
    """
    Build a comparison table from multiple model results.

    Parameters
    ----------
    results_dict : dict[str, dict]
        model_name → metrics dict

    Returns
    -------
    pd.DataFrame
    """
    df = pd.DataFrame(results_dict).T
    df.index.name = "Model"
    print("\n" + "=" * 60)
    print("MODEL COMPARISON")
    print("=" * 60)
    print(df.to_string())
    if save_path:
        df.to_csv(save_path)
        print(f"  ✓ Saved comparison → {save_path}")
    return df


# ══════════════════════════════════════════════════════════════════
# Diagnostic plots
# ══════════════════════════════════════════════════════════════════

def plot_roc_curves(results, save_path=None):
    """
    Plot ROC curves for multiple models.

    Parameters
    ----------
    results : dict[str, (y_true, y_prob)]
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    for name, (y_t, y_p) in results.items():
        fpr, tpr, _ = roc_curve(y_t, y_p)
        auc = roc_auc_score(y_t, y_p)
        ax.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
        print(f"  ✓ ROC plot → {save_path}")
    plt.close(fig)


def plot_pr_curves(results, save_path=None):
    """Plot Precision-Recall curves for multiple models."""
    fig, ax = plt.subplots(figsize=(8, 6))
    for name, (y_t, y_p) in results.items():
        prec, rec, _ = precision_recall_curve(y_t, y_p)
        ap = average_precision_score(y_t, y_p)
        ax.plot(rec, prec, label=f"{name} (AP={ap:.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curves")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
        print(f"  ✓ PR plot → {save_path}")
    plt.close(fig)


def reliability_diagram(y_true, y_prob, n_bins=10, save_path=None):
    """
    Reliability (calibration) diagram.
    Plots predicted probability vs observed frequency.
    """
    bins = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(y_prob, bins) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)

    bin_means = np.zeros(n_bins)
    bin_freqs = np.zeros(n_bins)
    bin_counts = np.zeros(n_bins)

    for i in range(n_bins):
        mask = bin_indices == i
        if mask.sum() > 0:
            bin_means[i]  = y_prob[mask].mean()
            bin_freqs[i]  = y_true[mask].mean()
            bin_counts[i] = mask.sum()

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8),
                                    gridspec_kw={"height_ratios": [3, 1]})

    # Calibration curve
    ax1.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Perfect calibration")
    ax1.bar(bin_means, bin_freqs, width=1.0/n_bins, alpha=0.4,
            edgecolor="steelblue", color="steelblue")
    ax1.plot(bin_means, bin_freqs, "o-", color="steelblue")
    ax1.set_xlabel("Mean predicted probability")
    ax1.set_ylabel("Observed frequency")
    ax1.set_title("Reliability Diagram")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Histogram of predictions
    ax2.bar(bin_means, bin_counts, width=1.0/n_bins, alpha=0.4,
            color="coral", edgecolor="coral")
    ax2.set_xlabel("Mean predicted probability")
    ax2.set_ylabel("Count")
    ax2.set_title("Prediction Distribution")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
        print(f"  ✓ Reliability diagram → {save_path}")
    plt.close(fig)


def performance_by_strata(y_true, y_prob, strata, strata_name="strata",
                          threshold=0.5, save_path=None):
    """
    Compute hit rate and false-alarm rate by classes of a stratification
    variable (e.g., slope bins or NDVI bins).
    """
    bins = np.quantile(strata, [0, 0.25, 0.5, 0.75, 1.0])
    bin_labels = [f"Q{i+1}" for i in range(4)]
    strata_binned = pd.cut(strata, bins=bins, labels=bin_labels,
                           include_lowest=True)

    y_pred = (y_prob >= threshold).astype(int)

    records = []
    for label in bin_labels:
        mask = strata_binned == label
        if mask.sum() == 0:
            continue
        yt = y_true[mask]
        yp = y_pred[mask]
        hits = ((yp == 1) & (yt == 1)).sum()
        false_alarms = ((yp == 1) & (yt == 0)).sum()
        misses = ((yp == 0) & (yt == 1)).sum()
        correct_neg = ((yp == 0) & (yt == 0)).sum()
        hit_rate = hits / (hits + misses) if (hits + misses) > 0 else 0
        fa_rate = false_alarms / (false_alarms + correct_neg) if (false_alarms + correct_neg) > 0 else 0
        records.append({
            "Stratum": label,
            "N": mask.sum(),
            "Hit Rate": hit_rate,
            "False Alarm Rate": fa_rate,
        })

    df = pd.DataFrame(records)
    print(f"\n  Performance by {strata_name}:")
    print(df.to_string(index=False))

    if save_path:
        df.to_csv(save_path, index=False)
    return df


# ── Self-test ─────────────────────────────────────────────────────
if __name__ == "__main__":
    np.random.seed(0)
    y_true = np.random.choice([0, 1], 200, p=[0.9, 0.1])
    y_prob = np.clip(y_true * 0.7 + np.random.rand(200) * 0.3, 0, 1)
    m = compute_metrics(y_true, y_prob)
    print_metrics("test", m)
    print("Evaluate self-test passed ✓")
