#!/usr/bin/env python3
"""
Model training & evaluation pipeline (enhanced).

Trains FIVE models:
  1. Physics-guided logistic regression (R_phys only)
  2. Attribute-only logistic regression (9 raw features)
  3. Random Forest (9 raw features)
  4. XGBoost gradient boosting (9 raw features)
  5. Weather-only logistic regression (T, RH, U only)

Runs advanced analyses:
  • Confusion matrices
  • SHAP values (RF, XGBoost)
  • Correlation heatmap
  • Threshold analysis (F1 vs threshold)
  • Seasonal performance breakdown
  • Cross-validated physics weight tuning

Usage:
    python run_train.py
"""

import os, sys, yaml
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from src.dataset import build_dataset, get_spatial_train_val_test, get_Xy
from src.models import (
    train_physics_logistic, train_attribute_logistic,
    train_random_forest, train_xgboost, train_weather_logistic,
    save_model, predict_proba, balance_dataset,
)
from src.fwi import calculate_fwi
from src.evaluate import (
    compute_metrics, print_metrics, compare_models,
    plot_roc_curves, plot_pr_curves, reliability_diagram,
    performance_by_strata,
)
from src.visualize import (
    plot_feature_distributions, plot_model_comparison_bar,
    plot_R_phys_timeseries,
)
from src.analysis import (
    plot_confusion_matrix, plot_shap_values,
    plot_correlation_heatmap, plot_threshold_analysis,
    seasonal_analysis, compute_morans_I,
)


def main():
    # ── Load config ─────────────────────────────────────────────
    with open(os.path.join(ROOT, "config.yaml")) as f:
        cfg = yaml.safe_load(f)

    features_dir = os.path.join(ROOT, cfg["paths"]["features"])
    models_dir   = os.path.join(ROOT, cfg["paths"]["models"])
    outputs_dir  = os.path.join(ROOT, cfg["paths"]["outputs"])
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(outputs_dir, exist_ok=True)

    # ── Step 1: Load dataset ───────────────────────────────────
    print("=" * 60)
    print("Step 1 · Loading feature dataset …")
    print("=" * 60)

    df = build_dataset(features_dir)

    # ── Step 2: Spatial Train / Val / Test split ───────────────────────
    print("\n" + "=" * 60)
    print("Step 2 · Spatial block split …")
    print("=" * 60)

    train_df, val_df, test_df = get_spatial_train_val_test(df, n_splits=5)
    
    # Calculate FWI Proxy baseline feature
    for subset in [train_df, val_df, test_df]:
        if set(["t_max", "rh_min", "u10_max", "p_tot"]).issubset(subset.columns):
            subset["FWI_proxy"] = calculate_fwi(
                temp=subset["t_max"], rh=subset["rh_min"],
                wind_kph=subset["u10_max"] * 3.6, precip=subset["p_tot"]
            )
        else:
            # Fallback
            subset["FWI_proxy"] = calculate_fwi(
                temp=subset["t_max"] if "t_max" in subset else np.zeros(len(subset)), 
                rh=subset["rh_min"] if "rh_min" in subset else np.zeros(len(subset)),
                wind_kph=(subset["u10_max"]*3.6) if "u10_max" in subset else np.zeros(len(subset)), 
                precip=np.zeros(len(subset))
            )

    # ── Step 3: Correlation heatmap ────────────────────────────
    print("\n" + "=" * 60)
    print("Step 3 · Correlation heatmap & feature distributions …")
    print("=" * 60)

    raw_features = ["t_max", "rh_min", "u10_max", "sm_top",
                    "ndvi", "ndwi", "slope", "frp_hist", "count_hist"]
    available_raw = [f for f in raw_features if f in df.columns]

    # Weather-only subset
    weather_features = ["t_max", "rh_min", "u10_max"]
    available_weather = [f for f in weather_features if f in df.columns]

    plot_correlation_heatmap(
        train_df, available_raw + ["R_phys", "ignition"],
        save_path=os.path.join(outputs_dir, "correlation_heatmap.png"),
    )

    plot_feature_distributions(
        train_df,
        save_path=os.path.join(outputs_dir, "feature_distributions.png"),
    )

    # ── Step 4: Train models ───────────────────────────────────
    print("\n" + "=" * 60)
    print("Step 4 · Training models …")
    print("=" * 60)

    results = {}
    preds   = {}

    # --- Model 1: Physics-guided logistic ---
    print("\n─── 1. Physics-Guided Logistic Regression ───")
    X_train, y_train = get_Xy(train_df, "R_phys")
    X_test, y_test   = get_Xy(test_df, "R_phys")
    
    X_train_bal, y_train_bal = balance_dataset(X_train, y_train)
    
    model_phys = train_physics_logistic(X_train_bal, y_train_bal)
    y_prob_phys = predict_proba(model_phys, X_test)
    results["Physics Logistic"] = compute_metrics(y_test, y_prob_phys)
    preds["Physics Logistic"] = (y_test, y_prob_phys)
    print_metrics("Physics Logistic", results["Physics Logistic"])
    save_model(model_phys, models_dir, "physics_logistic")

    # --- Model 2: Attribute-only logistic ---
    if available_raw:
        print("\n─── 2. Attribute-Only Logistic Regression ───")
        X_train_a, y_train_a = get_Xy(train_df, available_raw)
        X_test_a, y_test_a   = get_Xy(test_df, available_raw)
        X_train_a = np.nan_to_num(X_train_a, nan=0.0)
        X_test_a  = np.nan_to_num(X_test_a, nan=0.0)

        X_train_a_bal, y_train_a_bal = balance_dataset(X_train_a, y_train_a)

        model_attr = train_attribute_logistic(X_train_a_bal, y_train_a_bal)
        y_prob_attr = predict_proba(model_attr, X_test_a)
        results["Attribute Logistic"] = compute_metrics(y_test_a, y_prob_attr)
        preds["Attribute Logistic"] = (y_test_a, y_prob_attr)
        print_metrics("Attribute Logistic", results["Attribute Logistic"])
        save_model(model_attr, models_dir, "attribute_logistic")

        # --- Model 3: Random Forest ---
        print("\n─── 3. Random Forest Baseline ───")
        model_rf = train_random_forest(X_train_a_bal, y_train_a_bal)
        y_prob_rf = predict_proba(model_rf, X_test_a)
        results["Random Forest"] = compute_metrics(y_test_a, y_prob_rf)
        preds["Random Forest"] = (y_test_a, y_prob_rf)
        print_metrics("Random Forest", results["Random Forest"])
        save_model(model_rf, models_dir, "random_forest")

        # --- Model 4: XGBoost ---
        print("\n─── 4. XGBoost Gradient Boosting ───")
        model_xgb = train_xgboost(X_train_a_bal, y_train_a_bal)
        y_prob_xgb = predict_proba(model_xgb, X_test_a)
        results["XGBoost"] = compute_metrics(y_test_a, y_prob_xgb)
        preds["XGBoost"] = (y_test_a, y_prob_xgb)
        print_metrics("XGBoost", results["XGBoost"])
        save_model(model_xgb, models_dir, "xgboost")

    # --- Model 5: Weather-only / FWI Proxy Baseline ---
    if available_weather:
        print("\n─── 5. Weather-Only / FWI Proxy Baseline ───")
        feature_col = "FWI_proxy" if "FWI_proxy" in train_df.columns else available_weather
        X_train_w, y_train_w = get_Xy(train_df, feature_col)
        X_test_w, y_test_w   = get_Xy(test_df, feature_col)
        X_train_w = np.nan_to_num(X_train_w, nan=0.0)
        X_test_w  = np.nan_to_num(X_test_w, nan=0.0)

        X_train_w_bal, y_train_w_bal = balance_dataset(X_train_w, y_train_w)

        model_weather = train_weather_logistic(X_train_w_bal, y_train_w_bal)
        y_prob_weather = predict_proba(model_weather, X_test_w)
        results["FWI Proxy (Weather)"] = compute_metrics(y_test_w, y_prob_weather)
        preds["FWI Proxy (Weather)"] = (y_test_w, y_prob_weather)
        print_metrics("FWI Proxy (Weather)", results["FWI Proxy (Weather)"])
        save_model(model_weather, models_dir, "fwi_baseline")

    # --- Ablation Studies ---
    print("\n─── 6. Ablation Studies ───")
    ablation_vars = {"No Dryness": ["F_avail", "G_spread", "H_history"],
                     "No Spread": ["F_avail", "F_dry", "H_history"],
                     "No History": ["F_avail", "F_dry", "G_spread"]}
    
    for ab_name, ab_feats in ablation_vars.items():
        avail = [f for f in ab_feats if f in train_df.columns]
        if len(avail) == 3:
            train_ablated = train_df[avail].prod(axis=1).values.reshape(-1, 1)
            test_ablated = test_df[avail].prod(axis=1).values.reshape(-1, 1)
            
            X_ab_bal, y_ab_bal = balance_dataset(train_ablated, y_train)
            model_ab = train_physics_logistic(X_ab_bal, y_ab_bal)
            y_prob_ab = predict_proba(model_ab, test_ablated)
            
            res_name = f"Physics ({ab_name})"
            results[res_name] = compute_metrics(y_test, y_prob_ab)
            preds[res_name] = (y_test, y_prob_ab)
            print_metrics(res_name, results[res_name])

    # ── Step 5: Model comparison ───────────────────────────────
    print("\n" + "=" * 60)
    print("Step 5 · Model comparison …")
    print("=" * 60)

    comp_df = compare_models(
        results,
        save_path=os.path.join(outputs_dir, "model_comparison.csv"),
    )

    plot_roc_curves(preds, save_path=os.path.join(outputs_dir, "roc_curves.png"))
    plot_pr_curves(preds, save_path=os.path.join(outputs_dir, "pr_curves.png"))
    plot_model_comparison_bar(comp_df,
        save_path=os.path.join(outputs_dir, "model_comparison_bar.png"))

    # ── Step 6: Confusion matrices ─────────────────────────────
    print("\n" + "=" * 60)
    print("Step 6 · Confusion matrices …")
    print("=" * 60)

    for name, (y_t, y_p) in preds.items():
        safe_name = name.lower().replace(" ", "_")
        plot_confusion_matrix(
            y_t, y_p, model_name=name,
            save_path=os.path.join(outputs_dir, f"confusion_{safe_name}.png"),
        )

    # ── Step 7: Threshold analysis ─────────────────────────────
    print("\n" + "=" * 60)
    print("Step 7 · Threshold analysis …")
    print("=" * 60)

    best_t, best_f1 = plot_threshold_analysis(
        y_test, y_prob_phys, model_name="Physics Logistic",
        save_path=os.path.join(outputs_dir, "threshold_analysis.png"),
    )
    print(f"  Physics Logistic: optimal threshold={best_t:.2f}, F1={best_f1:.3f}")

    # ── Step 8: Reliability diagram ────────────────────────────
    print("\n" + "=" * 60)
    print("Step 8 · Reliability diagram …")
    print("=" * 60)

    reliability_diagram(
        y_test, y_prob_phys,
        save_path=os.path.join(outputs_dir, "reliability_diagram.png"),
    )

    # ── Step 9: SHAP values ────────────────────────────────────
    print("\n" + "=" * 60)
    print("Step 9 · SHAP analysis …")
    print("=" * 60)

    if available_raw:
        try:
            print("\n  [Random Forest SHAP]")
            plot_shap_values(
                model_rf, X_test_a, available_raw,
                model_name="Random Forest",
                save_path=os.path.join(outputs_dir, "shap_random_forest.png"),
            )

            print("  [XGBoost SHAP]")
            plot_shap_values(
                model_xgb, X_test_a, available_raw,
                model_name="XGBoost",
                save_path=os.path.join(outputs_dir, "shap_xgboost.png"),
            )
        except Exception as e:
            print(f"  ⚠ SHAP analysis skipped: {e}")

    # ── Step 10: Seasonal analysis ─────────────────────────────
    print("\n" + "=" * 60)
    print("Step 10 · Seasonal analysis …")
    print("=" * 60)

    if "time" in test_df.columns:
        monthly_df, seasonal_df = seasonal_analysis(
            test_df, y_prob_phys,
            model_name="Physics Logistic",
            save_path=os.path.join(outputs_dir, "seasonal_analysis.png"),
        )
        print("\n  Monthly breakdown:")
        print(monthly_df.to_string(index=False))
        print("\n  Seasonal breakdown:")
        print(seasonal_df.to_string())

    # ── Step 11: Performance by strata ─────────────────────────
    print("\n" + "=" * 60)
    print("Step 11 · Stratified performance …")
    print("=" * 60)

    if "slope" in test_df.columns:
        slope_vals = test_df["slope"].values
        if not np.all(np.isnan(slope_vals)):
            performance_by_strata(
                y_test, y_prob_phys, slope_vals,
                strata_name="Slope",
                save_path=os.path.join(outputs_dir, "performance_by_slope.csv"),
            )

    if "ndvi" in test_df.columns:
        ndvi_vals = test_df["ndvi"].values
        if not np.all(np.isnan(ndvi_vals)):
            performance_by_strata(
                y_test, y_prob_phys, ndvi_vals,
                strata_name="NDVI",
                save_path=os.path.join(outputs_dir, "performance_by_ndvi.csv"),
            )

    # ── Step 12: R_phys timeseries ─────────────────────────────
    print("\n" + "=" * 60)
    print("Step 12 · R_phys timeseries …")
    print("=" * 60)

    if "time" in test_df.columns:
        ts = test_df.groupby("time").agg(
            R_phys_mean=("R_phys", "mean"),
            fire_rate=("ignition", "mean"),
        ).reset_index()
        plot_R_phys_timeseries(
            ts["time"], ts["R_phys_mean"], ts["fire_rate"],
            save_path=os.path.join(outputs_dir, "R_phys_timeseries.png"),
        )

    print("\n" + "=" * 60)
    print("✅ Training & evaluation complete!")
    print("=" * 60)
    print(f"   Models saved to:  {models_dir}/")
    print(f"   Outputs saved to: {outputs_dir}/")
    print(f"\n   Generated {len(results)} model comparisons:")
    for name in results:
        print(f"     • {name}")
    print(f"\n   Generated plots:")
    for f in sorted(os.listdir(outputs_dir)):
        if f.endswith((".png", ".csv")):
            print(f"     • {f}")


if __name__ == "__main__":
    main()
