"""
Ablation study - systematically removes one physics factor at a time
from R_phys to measure each component's contribution.

Ablation variants:
  Full:        R = T~ . F_avail . F_dry . G_spread + H_history
  No F_avail:  R = T~ . 1.0    . F_dry . G_spread + H_history
  No F_dry:    R = T~ . F_avail . 1.0  . G_spread + H_history
  No G_spread: R = T~ . F_avail . F_dry . 1.0     + H_history
  No H_history:R = T~ . F_avail . F_dry . G_spread + 0.0
  No T:        R = 1  . F_avail . F_dry . G_spread + H_history
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, log_loss

from src.features import (
    compute_F_avail, compute_F_dry, compute_G_spread,
    compute_H_history, compute_R_phys,
)


def run_ablation(normed_data, y_train, y_test, train_mask, test_mask, cfg):
    """
    Run ablation study: train logistic regression on R_phys with each
    factor removed one at a time.

    Parameters
    ----------
    normed_data : dict of str -> flat arrays
    y_train, y_test : 1D arrays
    train_mask, test_mask : bool arrays (for indexing flat data)
    cfg : dict with alpha, beta, gamma

    Returns
    -------
    results : dict of variant_name -> metrics dict
    """
    alpha = cfg["alpha"]
    beta = cfg["beta"]
    gamma = cfg["gamma"]

    # Compute base factors
    F_a  = compute_F_avail(normed_data["ndvi"])
    F_d  = compute_F_dry(normed_data["ndwi"], normed_data["rh_min"],
                         normed_data["sm_top"], alpha)
    G_s  = compute_G_spread(normed_data["u10_max"], normed_data["slope"], beta)
    H_h  = compute_H_history(normed_data["frp_hist"],
                             normed_data["count_hist"], gamma)
    T_n  = normed_data["t_max"]

    # Define ablation variants
    variants = {
        "Full R_phys":           lambda: compute_R_phys(T_n, F_a, F_d, G_s, H_h),
        "No Temperature (T~=1)": lambda: compute_R_phys(np.ones_like(T_n), F_a, F_d, G_s, H_h),
        "No F_avail (=1)":       lambda: compute_R_phys(T_n, np.ones_like(F_a), F_d, G_s, H_h),
        "No F_dry (=1)":         lambda: compute_R_phys(T_n, F_a, np.ones_like(F_d), G_s, H_h),
        "No G_spread (=1)":      lambda: compute_R_phys(T_n, F_a, F_d, np.ones_like(G_s), H_h),
        "No H_history (=0)":     lambda: compute_R_phys(T_n, F_a, F_d, G_s, np.zeros_like(H_h)),
    }

    results = {}
    for name, compute_fn in variants.items():
        R = np.asarray(compute_fn()).flatten()

        R_train = R[train_mask].reshape(-1, 1)
        R_test  = R[test_mask].reshape(-1, 1)

        model = LogisticRegression(
            solver="lbfgs", max_iter=1000, class_weight="balanced"
        )
        model.fit(R_train, y_train.astype(int))
        y_prob = model.predict_proba(R_test)[:, 1]

        try:
            auc_roc = roc_auc_score(y_test, y_prob)
        except ValueError:
            auc_roc = np.nan
        try:
            auc_pr = average_precision_score(y_test, y_prob)
        except ValueError:
            auc_pr = np.nan
        try:
            logloss = log_loss(y_test, y_prob)
        except ValueError:
            logloss = np.nan

        results[name] = {
            "auc_roc": auc_roc,
            "auc_pr": auc_pr,
            "log_loss": logloss,
        }
        print(f"  {name:25s}  AUC-ROC={auc_roc:.4f}  AUC-PR={auc_pr:.4f}  "
              f"LogLoss={logloss:.4f}")

    return results


def plot_ablation(results, save_path=None):
    """
    Bar chart showing AUC-ROC drop when each factor is removed.
    The 'drop' from full R_phys quantifies each component's importance.
    """
    df = pd.DataFrame(results).T
    df.index.name = "Variant"
    full_auc = df.loc["Full R_phys", "auc_roc"]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # AUC-ROC bars
    colors = ["#2e7d32" if v == "Full R_phys" else "#e53935"
              for v in df.index]
    bars = axes[0].barh(df.index, df["auc_roc"], color=colors, alpha=0.7,
                        edgecolor="black")
    axes[0].axvline(full_auc, color="#2e7d32", linestyle="--", linewidth=2,
                    alpha=0.5, label=f"Full R_phys ({full_auc:.3f})")
    axes[0].set_xlabel("AUC-ROC", fontsize=12)
    axes[0].set_title("Ablation Study — AUC-ROC", fontsize=14)
    axes[0].legend()
    axes[0].grid(True, alpha=0.2, axis="x")

    # Add value labels
    for bar, val in zip(bars, df["auc_roc"]):
        axes[0].text(val + 0.003, bar.get_y() + bar.get_height() / 2,
                     f"{val:.4f}", va="center", fontsize=10)

    # Delta bars (drop from full)
    delta = full_auc - df["auc_roc"]
    delta = delta.drop("Full R_phys")
    delta_sorted = delta.sort_values(ascending=False)

    axes[1].barh(delta_sorted.index, delta_sorted.values,
                 color="#ff9800", alpha=0.7, edgecolor="black")
    axes[1].set_xlabel("ΔAUC-ROC (drop from full)", fontsize=12)
    axes[1].set_title("Component Importance (bigger = more important)", fontsize=14)
    axes[1].grid(True, alpha=0.2, axis="x")

    for i, (name, val) in enumerate(delta_sorted.items()):
        axes[1].text(val + 0.001, i, f"{val:+.4f}", va="center", fontsize=10)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  [OK] Ablation plot -> {save_path}")
    plt.close(fig)

    return df
