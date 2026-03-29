#!/usr/bin/env python3
"""
Retrain and compare baseline vs optimized physics logistic models.

- Keeps existing baseline model artifact untouched.
- Saves new models with explicit names.
- Writes metrics comparison on validation and test years.
"""

import json
import os
import shutil
import sys
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import xarray as xr
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.formula_profile import baseline_profile_from_cfg, merge_profile


FEATURES = [
    "t_max", "rh_min", "u10_max", "sm_top", "ndvi", "ndwi", "slope", "frp_hist", "count_hist",
]


def _align_flat_length(arr, target_len):
    arr = np.asarray(arr).reshape(-1)
    if arr.size == target_len:
        return arr
    if arr.size > 0 and target_len % arr.size == 0:
        return np.tile(arr, target_len // arr.size)
    raise ValueError(f"Cannot align array of length {arr.size} to {target_len}")


def _flatten(ds):
    target_len = ds["t_max"].values.reshape(-1).size
    flat = {k: _align_flat_length(ds[k].values, target_len) for k in FEATURES}
    y = _align_flat_length(ds["ignition"].values, target_len).astype(int)

    times = pd.to_datetime(ds.time.values)
    years = np.repeat(times.year.values, len(ds.latitude) * len(ds.longitude))
    return flat, y, years


def _compute_r(flat, profile):
    a = profile["alpha"]
    b = profile["beta"]
    g = profile["gamma"]
    mix = float(profile.get("mix_history", 0.0))

    f_avail = flat["ndvi"]
    f_dry = a["alpha1"] * (1 - flat["ndwi"]) + a["alpha2"] * (1 - flat["rh_min"]) + a["alpha3"] * (1 - flat["sm_top"])
    g_spread = 1 + b["beta1"] * flat["u10_max"] + b["beta2"] * flat["slope"]
    h_history = g["gamma1"] * flat["frp_hist"] + g["gamma2"] * flat["count_hist"]

    core = flat["t_max"] * f_avail * f_dry * g_spread
    return (1.0 - mix) * (core + h_history) + mix * (core * (1.0 + h_history))


def _eval(y_true, y_prob):
    return {
        "auc_roc": float(roc_auc_score(y_true, y_prob)),
        "auc_pr": float(average_precision_score(y_true, y_prob)),
        "brier": float(brier_score_loss(y_true, y_prob)),
        "log_loss": float(log_loss(y_true, y_prob, labels=[0, 1])),
    }


def main():
    with open(os.path.join(ROOT, "config.yaml"), "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    models_dir = os.path.join(ROOT, cfg["paths"]["models"])
    outputs_dir = os.path.join(ROOT, cfg["paths"]["outputs"])
    os.makedirs(outputs_dir, exist_ok=True)

    # Backup current baseline artifact before retraining outputs are created.
    baseline_artifact = os.path.join(models_dir, "physics_logistic.joblib")
    if os.path.exists(baseline_artifact):
        backup_dir = os.path.join(models_dir, "archive")
        os.makedirs(backup_dir, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        shutil.copy2(
            baseline_artifact,
            os.path.join(backup_dir, f"physics_logistic_before_profile_retrain_{stamp}.joblib"),
        )

    ds = xr.open_dataset(os.path.join(ROOT, cfg["paths"]["features"], "features.nc"))
    flat, y, years = _flatten(ds)
    ds.close()

    finite = np.isfinite(y)
    for k in FEATURES:
        finite &= np.isfinite(flat[k])

    train_mask = np.isin(years, cfg["time"]["train_years"]) & finite
    val_mask = np.isin(years, cfg["time"]["val_years"]) & finite
    test_mask = np.isin(years, cfg["time"]["test_years"]) & finite

    baseline_profile = baseline_profile_from_cfg(cfg)

    opt_profile = baseline_profile
    opt_payload_path = os.path.join(ROOT, "outputs", "optimized_formula_profile.json")
    if os.path.exists(opt_payload_path):
        with open(opt_payload_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        opt_profile = merge_profile(cfg, payload.get("optimized", {}).get("profile", {}))

    rows = []
    for name, profile in [("baseline", baseline_profile), ("optimized", opt_profile)]:
        r = _compute_r(flat, profile)

        x_train = r[train_mask].reshape(-1, 1)
        y_train = y[train_mask]
        x_val = r[val_mask].reshape(-1, 1)
        y_val = y[val_mask]
        x_test = r[test_mask].reshape(-1, 1)
        y_test = y[test_mask]

        model = LogisticRegression(solver="lbfgs", max_iter=1000, class_weight="balanced")
        model.fit(x_train, y_train)

        p_val = model.predict_proba(x_val)[:, 1]
        p_test = model.predict_proba(x_test)[:, 1]

        val_metrics = _eval(y_val, p_val)
        test_metrics = _eval(y_test, p_test)

        model_name = "physics_logistic_baseline_retrained.joblib" if name == "baseline" else "physics_logistic_optimized.joblib"
        joblib.dump(model, os.path.join(models_dir, model_name), compress=3)

        rows.append({"profile": name, "split": "val", **val_metrics})
        rows.append({"profile": name, "split": "test", **test_metrics})

    comp = pd.DataFrame(rows)
    comp_path = os.path.join(outputs_dir, "physics_profile_retrain_comparison.csv")
    comp.to_csv(comp_path, index=False)

    print("Retrain complete")
    print(f"  Saved: {os.path.join(models_dir, 'physics_logistic_baseline_retrained.joblib')}")
    print(f"  Saved: {os.path.join(models_dir, 'physics_logistic_optimized.joblib')}")
    print(f"  Comparison: {comp_path}")


if __name__ == "__main__":
    main()
