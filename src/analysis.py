"""
Advanced analysis – SHAP values, confusion matrix, correlation heatmap,
threshold analysis, seasonal breakdown, cross-validated weight tuning,
and spatial autocorrelation.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, f1_score, precision_score, recall_score,
)


# ══════════════════════════════════════════════════════════════════
# 1. Confusion Matrix Heatmap
# ══════════════════════════════════════════════════════════════════

def plot_confusion_matrix(y_true, y_prob, model_name="Model",
                          threshold=0.5, save_path=None):
    """
    Plot confusion matrix as a styled heatmap.
    """
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        cm, annot=True, fmt=",d", cmap="Blues",
        xticklabels=["No Fire", "Fire"],
        yticklabels=["No Fire", "Fire"],
        ax=ax, linewidths=0.5, linecolor="white",
        annot_kws={"size": 14},
    )
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("Actual", fontsize=12)
    ax.set_title(f"Confusion Matrix — {model_name}\n(threshold={threshold})",
                 fontsize=13)

    # Add precision/recall annotations
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec  = recall_score(y_true, y_pred, zero_division=0)
    f1   = f1_score(y_true, y_pred, zero_division=0)
    ax.text(0.5, -0.12, f"Precision={prec:.3f}  Recall={rec:.3f}  F1={f1:.3f}",
            transform=ax.transAxes, ha="center", fontsize=11, style="italic")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  ✓ Confusion matrix → {save_path}")
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════
# 2. SHAP Values
# ══════════════════════════════════════════════════════════════════

def plot_shap_values(model, X, feature_names, model_name="Model",
                     max_display=10, save_path=None):
    """
    Compute and plot SHAP values for a tree-based model.
    """
    import shap

    # Use a smaller sample for speed
    n_sample = min(5000, len(X))
    idx = np.random.choice(len(X), n_sample, replace=False)
    X_sample = X[idx]

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    # For binary classifiers, shap_values may be a list
    if isinstance(shap_values, list):
        shap_values = shap_values[1]  # class-1

    fig, axes = plt.subplots(1, 2, figsize=(18, 6))

    # Beeswarm-style summary
    plt.sca(axes[0])
    shap.summary_plot(
        shap_values, X_sample,
        feature_names=feature_names,
        max_display=max_display,
        show=False,
    )
    axes[0].set_title(f"SHAP Summary — {model_name}")

    # Bar chart of mean |SHAP|
    plt.sca(axes[1])
    shap.summary_plot(
        shap_values, X_sample,
        feature_names=feature_names,
        plot_type="bar",
        max_display=max_display,
        show=False,
    )
    axes[1].set_title(f"Mean |SHAP| — {model_name}")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  ✓ SHAP values → {save_path}")
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════
# 3. Correlation Heatmap
# ══════════════════════════════════════════════════════════════════

def plot_correlation_heatmap(df, features, save_path=None):
    """
    Plot correlation heatmap of normalised features.
    """
    available = [f for f in features if f in df.columns]
    corr = df[available].corr()

    fig, ax = plt.subplots(figsize=(10, 8))
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)

    cmap = sns.diverging_palette(250, 15, s=75, l=40, center="light", as_cmap=True)
    sns.heatmap(
        corr, mask=mask, annot=True, fmt=".2f",
        cmap=cmap, center=0, vmin=-1, vmax=1,
        ax=ax, linewidths=0.5,
        annot_kws={"size": 9},
    )
    ax.set_title("Feature Correlation Matrix", fontsize=14)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  ✓ Correlation heatmap → {save_path}")
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════
# 4. Probability Threshold Analysis
# ══════════════════════════════════════════════════════════════════

def plot_threshold_analysis(y_true, y_prob, model_name="Model",
                            save_path=None):
    """
    Plot F1, Precision, Recall vs probability threshold.
    Also marks the optimal F1 threshold.
    """
    thresholds = np.arange(0.01, 1.0, 0.01)
    f1s, precs, recs = [], [], []

    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        f1s.append(f1_score(y_true, y_pred, zero_division=0))
        precs.append(precision_score(y_true, y_pred, zero_division=0))
        recs.append(recall_score(y_true, y_pred, zero_division=0))

    best_idx = np.argmax(f1s)
    best_t = thresholds[best_idx]
    best_f1 = f1s[best_idx]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(thresholds, f1s, "b-", linewidth=2, label="F1 Score")
    ax.plot(thresholds, precs, "g--", linewidth=1.5, label="Precision")
    ax.plot(thresholds, recs, "r--", linewidth=1.5, label="Recall")

    ax.axvline(best_t, color="orange", linestyle=":", linewidth=2,
               label=f"Optimal threshold={best_t:.2f} (F1={best_f1:.3f})")

    ax.set_xlabel("Probability Threshold", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title(f"Threshold Analysis — {model_name}", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  ✓ Threshold analysis → {save_path}")
    plt.close(fig)

    return best_t, best_f1


# ══════════════════════════════════════════════════════════════════
# 5. Seasonal Performance Analysis
# ══════════════════════════════════════════════════════════════════

def seasonal_analysis(df, y_prob, model_name="Model", save_path=None):
    """
    Break down model performance by month and season.
    """
    from sklearn.metrics import roc_auc_score

    df = df.copy()
    df["y_prob"] = y_prob
    df["month"] = pd.to_datetime(df["time"]).dt.month

    # Define seasons
    season_map = {
        12: "Winter", 1: "Winter", 2: "Winter",
        3: "Spring", 4: "Spring", 5: "Spring",
        6: "Summer", 7: "Summer", 8: "Summer",
        9: "Fall", 10: "Fall", 11: "Fall",
    }
    df["season"] = df["month"].map(season_map)

    # Monthly analysis
    monthly = []
    for month in range(1, 13):
        mask = df["month"] == month
        subset = df[mask]
        if subset["ignition"].sum() > 0 and subset["ignition"].nunique() > 1:
            auc = roc_auc_score(subset["ignition"], subset["y_prob"])
        else:
            auc = np.nan
        monthly.append({
            "Month": month,
            "N_samples": len(subset),
            "Fire_rate": subset["ignition"].mean(),
            "AUC_ROC": auc,
        })
    monthly_df = pd.DataFrame(monthly)

    # Seasonal analysis
    seasonal = []
    for season in ["Winter", "Spring", "Summer", "Fall"]:
        mask = df["season"] == season
        subset = df[mask]
        if subset["ignition"].sum() > 0 and subset["ignition"].nunique() > 1:
            auc = roc_auc_score(subset["ignition"], subset["y_prob"])
        else:
            auc = np.nan
        seasonal.append({
            "Season": season,
            "N_samples": len(subset),
            "Fire_rate": subset["ignition"].mean() * 100,
            "AUC_ROC": auc,
        })
    seasonal_df = pd.DataFrame(seasonal)

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Monthly fire rate
    axes[0].bar(monthly_df["Month"], monthly_df["Fire_rate"] * 100,
                color="#e63946", alpha=0.7, edgecolor="black")
    axes[0].set_xlabel("Month")
    axes[0].set_ylabel("Fire Rate (%)")
    axes[0].set_title("Monthly Fire Rate")
    axes[0].set_xticks(range(1, 13))
    axes[0].grid(True, alpha=0.2, axis="y")

    # Monthly AUC
    valid_months = monthly_df.dropna(subset=["AUC_ROC"])
    axes[1].plot(valid_months["Month"], valid_months["AUC_ROC"],
                 "o-", color="#2196F3", linewidth=2, markersize=8)
    axes[1].axhline(0.5, color="gray", linestyle="--", alpha=0.5)
    axes[1].set_xlabel("Month")
    axes[1].set_ylabel("AUC-ROC")
    axes[1].set_title(f"Monthly AUC — {model_name}")
    axes[1].set_xticks(range(1, 13))
    axes[1].set_ylim(0.3, 1.0)
    axes[1].grid(True, alpha=0.2)

    # Seasonal summary
    season_order = ["Winter", "Spring", "Summer", "Fall"]
    seasonal_df = seasonal_df.set_index("Season").reindex(season_order)
    colors = ["#4FC3F7", "#81C784", "#FFB74D", "#E57373"]
    axes[2].bar(seasonal_df.index, seasonal_df["AUC_ROC"],
                color=colors, alpha=0.7, edgecolor="black")
    axes[2].axhline(0.5, color="gray", linestyle="--", alpha=0.5)
    axes[2].set_ylabel("AUC-ROC")
    axes[2].set_title(f"Seasonal AUC — {model_name}")
    axes[2].set_ylim(0.3, 1.0)
    axes[2].grid(True, alpha=0.2, axis="y")

    plt.suptitle(f"Seasonal Performance Analysis — {model_name}", fontsize=14)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  ✓ Seasonal analysis → {save_path}")
    plt.close(fig)

    return monthly_df, seasonal_df


# ══════════════════════════════════════════════════════════════════
# 6. Cross-Validated Weight Tuning
# ══════════════════════════════════════════════════════════════════

def tune_physics_weights(normed_data, firms_ignition, cfg, val_mask=None):
    """
    Grid search over α, β, γ weights to find the combination
    that maximises AUC-ROC on the validation set.

    Parameters
    ----------
    normed_data : dict of str → ndarray (flattened)
    firms_ignition : ndarray (flattened)
    cfg : dict
    val_mask : bool ndarray (which samples are validation)

    Returns
    -------
    best_params : dict with best alpha, beta, gamma
    results_df : DataFrame with all tried combinations
    """
    from sklearn.metrics import roc_auc_score
    from src.features import compute_F_avail, compute_F_dry, compute_G_spread, compute_H_history, compute_R_phys
    from sklearn.linear_model import LogisticRegression

    # Define search grid (lightweight)
    alpha_grid = [
        {"alpha1": 0.5, "alpha2": 0.3, "alpha3": 0.2},
        {"alpha1": 0.4, "alpha2": 0.3, "alpha3": 0.3},
        {"alpha1": 0.33, "alpha2": 0.34, "alpha3": 0.33},
        {"alpha1": 0.3, "alpha2": 0.4, "alpha3": 0.3},
        {"alpha1": 0.2, "alpha2": 0.3, "alpha3": 0.5},
    ]
    beta_grid = [
        {"beta1": 0.7, "beta2": 0.3},
        {"beta1": 0.6, "beta2": 0.4},
        {"beta1": 0.5, "beta2": 0.5},
        {"beta1": 0.4, "beta2": 0.6},
    ]
    gamma_grid = [
        {"gamma1": 0.7, "gamma2": 0.3},
        {"gamma1": 0.6, "gamma2": 0.4},
        {"gamma1": 0.5, "gamma2": 0.5},
    ]

    y = firms_ignition
    if val_mask is None:
        val_mask = np.ones(len(y), dtype=bool)

    best_auc = -1
    best_params = {}
    records = []

    total = len(alpha_grid) * len(beta_grid) * len(gamma_grid)
    count = 0

    for alpha in alpha_grid:
        for beta in beta_grid:
            for gamma in gamma_grid:
                count += 1
                F_a = compute_F_avail(normed_data["ndvi"])
                F_d = compute_F_dry(normed_data["ndwi"], normed_data["rh_min"],
                                    normed_data["sm_top"], alpha)
                G_s = compute_G_spread(normed_data["u10_max"],
                                       normed_data["slope"], beta)
                H_h = compute_H_history(normed_data["frp_hist"],
                                        normed_data["count_hist"], gamma)
                R = compute_R_phys(normed_data["t_max"], F_a, F_d, G_s, H_h)

                R_flat = np.asarray(R).flatten()
                y_flat = np.asarray(y).flatten()

                # Use only validation portion
                R_val = R_flat[val_mask]
                y_val = y_flat[val_mask]

                if y_val.sum() > 0 and len(np.unique(y_val)) > 1:
                    try:
                        model = LogisticRegression(
                            solver="lbfgs", max_iter=500,
                            class_weight="balanced"
                        )
                        model.fit(R_val.reshape(-1, 1), y_val.astype(int))
                        prob = model.predict_proba(R_val.reshape(-1, 1))[:, 1]
                        auc = roc_auc_score(y_val, prob)
                    except Exception:
                        auc = 0.0
                else:
                    auc = 0.0

                records.append({
                    **alpha, **beta, **gamma,
                    "auc_roc": auc,
                })

                if auc > best_auc:
                    best_auc = auc
                    best_params = {"alpha": alpha, "beta": beta, "gamma": gamma}

    results_df = pd.DataFrame(records).sort_values("auc_roc", ascending=False)
    print(f"\n  Weight tuning: tested {total} combinations")
    print(f"  Best AUC-ROC: {best_auc:.4f}")
    print(f"  Best α: {best_params['alpha']}")
    print(f"  Best β: {best_params['beta']}")
    print(f"  Best γ: {best_params['gamma']}")

    return best_params, results_df


# ══════════════════════════════════════════════════════════════════
# 7. Spatial Autocorrelation (Moran's I)
# ══════════════════════════════════════════════════════════════════

def compute_morans_I(residuals_grid):
    """
    Compute Moran's I for spatial autocorrelation of model residuals.
    Uses a simple queen contiguity on the grid.

    Parameters
    ----------
    residuals_grid : 2D ndarray (lat, lon) – mean residuals per cell

    Returns
    -------
    I : float – Moran's I statistic
    """
    from scipy.ndimage import uniform_filter

    r = residuals_grid.copy()
    r_mean = np.nanmean(r)
    r_centered = r - r_mean

    # Spatial lag via 3×3 neighbourhood average
    spatial_lag = uniform_filter(r_centered, size=3, mode="constant")

    numerator = np.nansum(r_centered * spatial_lag)
    denominator = np.nansum(r_centered ** 2)

    n = np.sum(~np.isnan(r))
    W = n  # approximate total weight

    if denominator > 0:
        I = (n / W) * (numerator / denominator)
    else:
        I = 0.0

    print(f"  Moran's I: {I:.4f}")
    print(f"    I ≈ 0 → no spatial autocorrelation")
    print(f"    I > 0 → positive autocorrelation (clustered errors)")
    print(f"    I < 0 → negative autocorrelation (dispersed errors)")

    return I
