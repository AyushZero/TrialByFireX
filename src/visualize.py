"""
Visualization - maps, curves, and diagnostic plots for the
ignition probability modelling project.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns


def plot_probability_map(prob_grid, lats, lons, date_str="",
                         save_path=None, title=None):
    """
    Plot ignition probability map with a colour-graded display.

    Parameters
    ----------
    prob_grid : 2D ndarray (n_lat, n_lon)
    lats, lons : 1D arrays
    date_str : str - for title
    """
    fig, ax = plt.subplots(figsize=(12, 8))

    # Custom colormap: green -> yellow -> orange -> red
    colors = ["#2d6a4f", "#95d5b2", "#ffff3f", "#ff9f1c", "#e63946"]
    cmap = mcolors.LinearSegmentedColormap.from_list("fire_risk", colors)

    im = ax.pcolormesh(
        lons, lats, prob_grid,
        cmap=cmap, vmin=0, vmax=1, shading="auto",
    )
    cb = fig.colorbar(im, ax=ax, shrink=0.7, label="Ignition Probability")

    if title:
        ax.set_title(title, fontsize=14)
    elif date_str:
        ax.set_title(f"Ignition Probability — {date_str}", fontsize=14)
    else:
        ax.set_title("Ignition Probability Map", fontsize=14)

    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_aspect("equal")

    # Add state outline hint
    ax.text(-119, 37, "California", fontsize=12, ha="center", alpha=0.5,
            style="italic")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  [OK] Probability map -> {save_path}")
    plt.close(fig)


def plot_feature_distributions(df, features=None, save_path=None):
    """
    Plot histograms of normalised features and R_phys,
    coloured by ignition label.
    """
    if features is None:
        features = ["t_max", "rh_min", "u10_max", "sm_top",
                     "ndvi", "ndwi", "slope", "R_phys"]

    n = len(features)
    ncols = 3
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    axes = axes.flatten()

    for i, feat in enumerate(features):
        if feat not in df.columns:
            continue
        ax = axes[i]
        for label, color in [(0, "#2196F3"), (1, "#F44336")]:
            subset = df[df["ignition"] == label][feat].dropna()
            ax.hist(subset, bins=50, alpha=0.5, color=color,
                    label=f"{'Fire' if label else 'No fire'}", density=True)
        ax.set_title(feat)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.2)

    # Hide unused axes
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle("Feature Distributions by Ignition Label", fontsize=14)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  [OK] Feature distributions -> {save_path}")
    plt.close(fig)


def plot_R_phys_timeseries(dates, R_phys_mean, y_mean=None, save_path=None):
    """
    Plot the spatial-mean R_phys over time, optionally overlaying
    observed fire rate.
    """
    fig, ax1 = plt.subplots(figsize=(14, 5))

    ax1.plot(dates, R_phys_mean, color="#ff6b35", label="R_phys (mean)")
    ax1.set_xlabel("Date")
    ax1.set_ylabel("R_phys (spatial mean)", color="#ff6b35")
    ax1.tick_params(axis="y", labelcolor="#ff6b35")

    if y_mean is not None:
        ax2 = ax1.twinx()
        ax2.fill_between(dates, y_mean, alpha=0.3, color="#e63946",
                         label="Fire rate")
        ax2.set_ylabel("Fire rate (spatial mean)", color="#e63946")
        ax2.tick_params(axis="y", labelcolor="#e63946")

    ax1.set_title("Temporal Evolution of Physics Risk Index")
    ax1.grid(True, alpha=0.2)
    fig.legend(loc="upper right", bbox_to_anchor=(0.95, 0.95))
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  [OK] R_phys timeseries -> {save_path}")
    plt.close(fig)


def plot_model_comparison_bar(comparison_df, save_path=None):
    """Bar chart comparing models across all metrics."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    metrics = ["auc_roc", "auc_pr", "brier_score"]
    titles  = ["AUC-ROC [UP]", "AUC-PR [UP]", "Brier Score [DOWN]"]
    colors  = ["#2196F3", "#4CAF50", "#FF9800"]

    for ax, metric, title, color in zip(axes, metrics, titles, colors):
        if metric in comparison_df.columns:
            comparison_df[metric].plot(kind="bar", ax=ax, color=color, alpha=0.7,
                                       edgecolor="black")
            ax.set_title(title)
            ax.set_ylabel(metric)
            ax.tick_params(axis="x", rotation=45)
            ax.grid(True, alpha=0.2, axis="y")

    plt.suptitle("Model Comparison", fontsize=14)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  [OK] Model comparison -> {save_path}")
    plt.close(fig)
