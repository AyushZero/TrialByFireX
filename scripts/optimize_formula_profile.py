#!/usr/bin/env python3
"""
Optimize physics formula profile (alpha/beta/gamma + mix_history).

Objective combines discrimination and calibration:
  score = w_pr * AUC-PR + w_roc * AUC-ROC - w_brier * Brier

Weights are constrained to sum to 1 per factor group:
  alpha1 + alpha2 + alpha3 = 1
  beta1 + beta2 = 1
  gamma1 + gamma2 = 1
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import xarray as xr
import yaml
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.formula_profile import baseline_profile_from_cfg, normalize_profile


FEATURES = [
    "t_max",
    "rh_min",
    "u10_max",
    "sm_top",
    "ndvi",
    "ndwi",
    "slope",
    "frp_hist",
    "count_hist",
]


def _align_flat_length(arr, target_len):
    arr = np.asarray(arr).reshape(-1)
    if arr.size == target_len:
        return arr
    if arr.size > 0 and target_len % arr.size == 0:
        return np.tile(arr, target_len // arr.size)
    raise ValueError(f"Cannot align array of length {arr.size} to {target_len}")


def _build_flat_data(ds, cfg):
    target_len = ds["t_max"].values.reshape(-1).size
    flat = {var: _align_flat_length(ds[var].values, target_len) for var in FEATURES}
    y = _align_flat_length(ds["ignition"].values, target_len).astype(int)

    times = pd.to_datetime(ds.time.values)
    n_lat = len(ds.latitude)
    n_lon = len(ds.longitude)
    years = np.repeat(times.year.values, n_lat * n_lon)

    train_mask = np.isin(years, cfg["time"]["train_years"])
    val_mask = np.isin(years, cfg["time"]["val_years"])

    finite = np.ones(target_len, dtype=bool)
    for var in FEATURES:
        finite &= np.isfinite(flat[var])
    finite &= np.isfinite(y)

    train_mask &= finite
    val_mask &= finite
    return flat, y, train_mask, val_mask


def _compute_r_phys(flat, params):
    a1, a2, a3, b1, b2, g1, g2, mix = params

    f_avail = flat["ndvi"]
    f_dry = a1 * (1.0 - flat["ndwi"]) + a2 * (1.0 - flat["rh_min"]) + a3 * (1.0 - flat["sm_top"])
    g_spread = 1.0 + b1 * flat["u10_max"] + b2 * flat["slope"]
    h_history = g1 * flat["frp_hist"] + g2 * flat["count_hist"]

    core = flat["t_max"] * f_avail * f_dry * g_spread
    return (1.0 - mix) * (core + h_history) + mix * (core * (1.0 + h_history))


def _evaluate(flat, y, train_mask, val_mask, params, w_pr, w_roc, w_brier):
    r = _compute_r_phys(flat, params)

    x_train = r[train_mask].reshape(-1, 1)
    x_val = r[val_mask].reshape(-1, 1)
    y_train = y[train_mask]
    y_val = y[val_mask]

    if x_train.shape[0] == 0 or x_val.shape[0] == 0 or np.unique(y_train).size < 2:
        return -1e6, {"auc_pr": np.nan, "auc_roc": np.nan, "brier": np.nan}

    model = LogisticRegression(solver="lbfgs", max_iter=1000, class_weight="balanced")
    model.fit(x_train, y_train)
    p_val = model.predict_proba(x_val)[:, 1]

    try:
        auc_pr = float(average_precision_score(y_val, p_val))
    except ValueError:
        auc_pr = np.nan
    try:
        auc_roc = float(roc_auc_score(y_val, p_val))
    except ValueError:
        auc_roc = np.nan
    brier = float(brier_score_loss(y_val, p_val))

    if not np.isfinite(auc_pr):
        auc_pr = 0.0
    if not np.isfinite(auc_roc):
        auc_roc = 0.5

    score = w_pr * auc_pr + w_roc * auc_roc - w_brier * brier
    return score, {"auc_pr": auc_pr, "auc_roc": auc_roc, "brier": brier}


def main():
    parser = argparse.ArgumentParser(description="Optimize formula profile")
    parser.add_argument("--sample-size", type=int, default=250000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--w-pr", type=float, default=1.0)
    parser.add_argument("--w-roc", type=float, default=0.2)
    parser.add_argument("--w-brier", type=float, default=0.2)
    parser.add_argument("--restarts", type=int, default=12)
    parser.add_argument("--out", default="outputs/optimized_formula_profile.json")
    args = parser.parse_args()

    np.random.seed(args.seed)

    with open(os.path.join(ROOT, "config.yaml"), "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    ds = xr.open_dataset(os.path.join(ROOT, cfg["paths"]["features"], "features.nc"))
    flat, y, train_mask, val_mask = _build_flat_data(ds, cfg)
    ds.close()

    train_idx = np.where(train_mask)[0]
    val_idx = np.where(val_mask)[0]

    if len(train_idx) == 0 or len(val_idx) == 0:
        raise RuntimeError("No valid train/val samples found for optimization")

    if len(train_idx) > args.sample_size:
        train_keep = np.random.choice(train_idx, size=args.sample_size, replace=False)
    else:
        train_keep = train_idx

    if len(val_idx) > args.sample_size:
        val_keep = np.random.choice(val_idx, size=args.sample_size, replace=False)
    else:
        val_keep = val_idx

    train_sub = np.zeros_like(train_mask)
    val_sub = np.zeros_like(val_mask)
    train_sub[train_keep] = True
    val_sub[val_keep] = True

    baseline = baseline_profile_from_cfg(cfg)
    x0 = np.array(
        [
            baseline["alpha"]["alpha1"],
            baseline["alpha"]["alpha2"],
            baseline["alpha"]["alpha3"],
            baseline["beta"]["beta1"],
            baseline["beta"]["beta2"],
            baseline["gamma"]["gamma1"],
            baseline["gamma"]["gamma2"],
            baseline["mix_history"],
        ],
        dtype=float,
    )

    bounds = [
        (0.05, 0.9),
        (0.05, 0.9),
        (0.05, 0.9),
        (0.05, 0.95),
        (0.05, 0.95),
        (0.05, 0.95),
        (0.05, 0.95),
        (0.0, 1.0),
    ]

    constraints = [
        {"type": "eq", "fun": lambda p: p[0] + p[1] + p[2] - 1.0},
        {"type": "eq", "fun": lambda p: p[3] + p[4] - 1.0},
        {"type": "eq", "fun": lambda p: p[5] + p[6] - 1.0},
    ]

    def objective(params):
        score, _ = _evaluate(flat, y, train_sub, val_sub, params, args.w_pr, args.w_roc, args.w_brier)
        return -score

    base_score, base_metrics = _evaluate(flat, y, train_sub, val_sub, x0, args.w_pr, args.w_roc, args.w_brier)

    history_rows = []
    best_params = x0.copy()
    best_score = base_score
    best_result = None

    starts = [x0.copy()]
    for _ in range(max(args.restarts - 1, 0)):
        a = np.random.dirichlet([1.0, 1.0, 1.0])
        b = np.random.dirichlet([1.0, 1.0])
        g = np.random.dirichlet([1.0, 1.0])
        mix = np.random.uniform(0.0, 1.0)
        starts.append(np.array([a[0], a[1], a[2], b[0], b[1], g[0], g[1], mix], dtype=float))

    for i, start in enumerate(starts, start=1):
        result = minimize(
            objective,
            start,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 150, "ftol": 1e-7, "disp": False},
        )
        cur_params = result.x
        cur_score, cur_metrics = _evaluate(
            flat, y, train_sub, val_sub, cur_params, args.w_pr, args.w_roc, args.w_brier
        )

        history_rows.append(
            {
                "restart": i,
                "success": bool(result.success),
                "iterations": int(result.nit),
                "score": float(cur_score),
                "auc_pr": float(cur_metrics["auc_pr"]),
                "auc_roc": float(cur_metrics["auc_roc"]),
                "brier": float(cur_metrics["brier"]),
            }
        )

        if cur_score > best_score:
            best_score = cur_score
            best_params = cur_params.copy()
            best_result = result

    opt_score, opt_metrics = _evaluate(flat, y, train_sub, val_sub, best_params, args.w_pr, args.w_roc, args.w_brier)

    optimized_profile = normalize_profile(
        {
            "name": "optimized_profile",
            "alpha": {"alpha1": best_params[0], "alpha2": best_params[1], "alpha3": best_params[2]},
            "beta": {"beta1": best_params[3], "beta2": best_params[4]},
            "gamma": {"gamma1": best_params[5], "gamma2": best_params[6]},
            "mix_history": float(best_params[7]),
        }
    )

    payload = {
        "objective": {
            "w_pr": args.w_pr,
            "w_roc": args.w_roc,
            "w_brier": args.w_brier,
            "sample_size": int(args.sample_size),
            "seed": int(args.seed),
            "restarts": int(args.restarts),
        },
        "optimization": {
            "success": bool(best_result.success) if best_result is not None else True,
            "message": str(best_result.message) if best_result is not None else "baseline retained",
            "iterations": int(best_result.nit) if best_result is not None else 0,
        },
        "baseline": {
            "profile": baseline,
            "score": float(base_score),
            **base_metrics,
        },
        "optimized": {
            "profile": optimized_profile,
            "score": float(opt_score),
            **opt_metrics,
        },
        "improvement": {
            "delta_score": float(opt_score - base_score),
            "delta_auc_pr": float(opt_metrics["auc_pr"] - base_metrics["auc_pr"]),
            "delta_auc_roc": float(opt_metrics["auc_roc"] - base_metrics["auc_roc"]),
            "delta_brier": float(opt_metrics["brier"] - base_metrics["brier"]),
        },
    }

    out_path = os.path.join(ROOT, args.out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    comp_path = os.path.join(ROOT, "outputs", "formula_profile_comparison.csv")
    pd.DataFrame(
        [
            {
                "profile": "baseline",
                "score": payload["baseline"]["score"],
                "auc_pr": payload["baseline"]["auc_pr"],
                "auc_roc": payload["baseline"]["auc_roc"],
                "brier": payload["baseline"]["brier"],
                "mix_history": payload["baseline"]["profile"]["mix_history"],
            },
            {
                "profile": "optimized",
                "score": payload["optimized"]["score"],
                "auc_pr": payload["optimized"]["auc_pr"],
                "auc_roc": payload["optimized"]["auc_roc"],
                "brier": payload["optimized"]["brier"],
                "mix_history": payload["optimized"]["profile"]["mix_history"],
            },
        ]
    ).to_csv(comp_path, index=False)

    history_path = os.path.join(ROOT, "outputs", "formula_profile_optimization_history.csv")
    pd.DataFrame(history_rows).to_csv(history_path, index=False)

    print("Optimization complete")
    print(f"  Output profile: {args.out}")
    print(f"  Baseline score:  {base_score:.6f}")
    print(f"  Optimized score: {opt_score:.6f}")
    print(f"  Delta score:     {opt_score - base_score:+.6f}")
    print(f"  History:         outputs/formula_profile_optimization_history.csv")


if __name__ == "__main__":
    main()
