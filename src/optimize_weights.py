"""
Weight optimization for physics-guided parameters (alpha, beta, gamma).

Uses scipy.optimize to find optimal weights that maximize AUC-ROC
on validation data.

The optimization tunes the physics formula weights:
  - alpha: [alpha1, alpha2, alpha3] for F_dry
  - beta: [beta1, beta2] for G_spread
  - gamma: [gamma1, gamma2] for H_history

Usage:
    from src.optimize_weights import optimize_weights
    optimized = optimize_weights(normed_data, y_train, y_val, train_mask, val_mask)
"""

import numpy as np
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from src.features import (
    compute_F_avail, compute_F_dry, compute_G_spread,
    compute_H_history, compute_R_phys
)


def _compute_R_with_weights(normed_data, params):
    """
    Compute R_phys with given weight parameters.

    Parameters
    ----------
    normed_data : dict of arrays
    params : array [alpha1, alpha2, alpha3, beta1, beta2, gamma1, gamma2]

    Returns
    -------
    R_phys : array
    """
    alpha = {"alpha1": params[0], "alpha2": params[1], "alpha3": params[2]}
    beta = {"beta1": params[3], "beta2": params[4]}
    gamma = {"gamma1": params[5], "gamma2": params[6]}

    F_a = compute_F_avail(normed_data["ndvi"])
    F_d = compute_F_dry(normed_data["ndwi"], normed_data["rh_min"],
                        normed_data["sm_top"], alpha)
    G_s = compute_G_spread(normed_data["u10_max"], normed_data["slope"], beta)
    H_h = compute_H_history(normed_data["frp_hist"],
                            normed_data["count_hist"], gamma)
    R = compute_R_phys(normed_data["t_max"], F_a, F_d, G_s, H_h)

    return np.asarray(R).flatten()


def _objective(params, normed_data, y_train, y_val, train_mask, val_mask):
    """
    Objective function: negative AUC-ROC (to minimize).

    Parameters
    ----------
    params : array [alpha1, alpha2, alpha3, beta1, beta2, gamma1, gamma2]
    normed_data : dict of arrays (full data, not masked)
    y_train, y_val : arrays of labels
    train_mask, val_mask : boolean masks for train/val splits

    Returns
    -------
    float : negative AUC-ROC
    """
    R = _compute_R_with_weights(normed_data, params)

    R_train = R[train_mask].reshape(-1, 1)
    R_val = R[val_mask].reshape(-1, 1)

    # Train logistic regression
    model = LogisticRegression(solver="lbfgs", max_iter=1000, class_weight="balanced")
    model.fit(R_train, y_train.astype(int))
    y_prob = model.predict_proba(R_val)[:, 1]

    try:
        auc = roc_auc_score(y_val, y_prob)
    except ValueError:
        auc = 0.5  # Default if only one class

    return -auc  # Minimize negative AUC


def optimize_weights(normed_data, y_train, y_val, train_mask, val_mask,
                     initial_weights=None, verbose=True):
    """
    Optimize alpha, beta, gamma weights using scipy.optimize.

    Parameters
    ----------
    normed_data : dict of arrays
        Keys: t_max, rh_min, u10_max, sm_top, ndvi, ndwi, slope, frp_hist, count_hist
    y_train : array
        Training labels
    y_val : array
        Validation labels
    train_mask : array
        Boolean mask for training samples
    val_mask : array
        Boolean mask for validation samples
    initial_weights : list, optional
        Starting weights [alpha1, alpha2, alpha3, beta1, beta2, gamma1, gamma2]
    verbose : bool
        Print optimization progress

    Returns
    -------
    dict : Optimized config dicts for alpha, beta, gamma
    """
    if initial_weights is None:
        # Default starting point from config.yaml
        initial_weights = [0.4, 0.3, 0.3, 0.6, 0.4, 0.6, 0.4]

    # Bounds for each weight (reasonable physical ranges)
    bounds = [
        (0.1, 0.8),  # alpha1 (NDWI)
        (0.1, 0.8),  # alpha2 (RH)
        (0.1, 0.8),  # alpha3 (SM)
        (0.2, 1.0),  # beta1 (wind)
        (0.1, 0.8),  # beta2 (slope)
        (0.2, 1.0),  # gamma1 (FRP history)
        (0.1, 0.8),  # gamma2 (count history)
    ]

    if verbose:
        print("  Optimizing physics weights (alpha, beta, gamma)...")
        print(f"    Initial weights: {initial_weights}")

    result = minimize(
        _objective,
        initial_weights,
        args=(normed_data, y_train, y_val, train_mask, val_mask),
        method='L-BFGS-B',
        bounds=bounds,
        options={'maxiter': 100, 'disp': verbose}
    )

    opt_params = result.x

    if verbose:
        print(f"\n  Optimization complete:")
        print(f"    alpha: [{opt_params[0]:.3f}, {opt_params[1]:.3f}, {opt_params[2]:.3f}]")
        print(f"    beta:  [{opt_params[3]:.3f}, {opt_params[4]:.3f}]")
        print(f"    gamma: [{opt_params[5]:.3f}, {opt_params[6]:.3f}]")
        print(f"    Best validation AUC-ROC: {-result.fun:.4f}")

    return {
        "alpha": {
            "alpha1": float(opt_params[0]),
            "alpha2": float(opt_params[1]),
            "alpha3": float(opt_params[2])
        },
        "beta": {
            "beta1": float(opt_params[3]),
            "beta2": float(opt_params[4])
        },
        "gamma": {
            "gamma1": float(opt_params[5]),
            "gamma2": float(opt_params[6])
        },
        "best_auc": float(-result.fun),
        "success": result.success,
    }


def grid_search_weights(normed_data, y_train, y_val, train_mask, val_mask,
                        n_samples=50, verbose=True):
    """
    Random grid search for weight optimization.

    Samples random weight combinations and evaluates them.
    Useful when the optimization landscape is non-smooth.

    Parameters
    ----------
    n_samples : int
        Number of random combinations to try

    Returns
    -------
    dict : Best weights found
    """
    np.random.seed(42)

    best_auc = 0
    best_params = None

    if verbose:
        print(f"  Grid search: testing {n_samples} random weight combinations...")

    for i in range(n_samples):
        # Random weights (summing constraints for alpha)
        alpha = np.random.dirichlet([1, 1, 1])  # Sum to 1
        beta = np.random.uniform(0.2, 0.8, 2)
        gamma = np.random.uniform(0.2, 0.8, 2)

        params = np.concatenate([alpha, beta, gamma])
        neg_auc = _objective(params, normed_data, y_train, y_val, train_mask, val_mask)
        auc = -neg_auc

        if auc > best_auc:
            best_auc = auc
            best_params = params
            if verbose:
                print(f"    [{i+1}/{n_samples}] New best AUC: {auc:.4f}")

    if verbose:
        print(f"\n  Grid search complete. Best AUC: {best_auc:.4f}")

    return {
        "alpha": {
            "alpha1": float(best_params[0]),
            "alpha2": float(best_params[1]),
            "alpha3": float(best_params[2])
        },
        "beta": {
            "beta1": float(best_params[3]),
            "beta2": float(best_params[4])
        },
        "gamma": {
            "gamma1": float(best_params[5]),
            "gamma2": float(best_params[6])
        },
        "best_auc": float(best_auc),
    }


def compare_with_defaults(normed_data, y_train, y_val, train_mask, val_mask, cfg):
    """
    Compare default weights from config with optimized weights.

    This is useful for demonstrating the value of weight optimization
    in the research paper.

    Parameters
    ----------
    cfg : dict
        Config with 'alpha', 'beta', 'gamma' sections

    Returns
    -------
    comparison : dict with 'default' and 'optimized' results
    """
    # Evaluate default weights
    default_alpha = cfg["alpha"]
    default_beta = cfg["beta"]
    default_gamma = cfg["gamma"]

    default_params = [
        default_alpha["alpha1"], default_alpha["alpha2"], default_alpha["alpha3"],
        default_beta["beta1"], default_beta["beta2"],
        default_gamma["gamma1"], default_gamma["gamma2"]
    ]
    default_auc = -_objective(default_params, normed_data, y_train, y_val,
                              train_mask, val_mask)

    # Optimize
    opt_weights = optimize_weights(normed_data, y_train, y_val,
                                   train_mask, val_mask, verbose=False)

    print("\n  Weight Comparison:")
    print(f"    Default weights AUC-ROC:   {default_auc:.4f}")
    print(f"    Optimized weights AUC-ROC: {opt_weights['best_auc']:.4f}")
    improvement = (opt_weights['best_auc'] - default_auc) * 100
    print(f"    Improvement: {improvement:+.2f}%")

    return {
        "default": {
            "auc": default_auc,
            "weights": {
                "alpha": default_alpha,
                "beta": default_beta,
                "gamma": default_gamma
            }
        },
        "optimized": opt_weights,
        "improvement_pct": improvement
    }


# ── Self-test ─────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Weight optimization module loaded successfully.")
    print("Usage: from src.optimize_weights import optimize_weights")
