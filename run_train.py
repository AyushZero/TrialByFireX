#!/usr/bin/env python3
"""
Model training & evaluation pipeline (conference-ready version).

Models (7 total):
  1. Physics-guided logistic regression (R_phys only)
  2. Attribute-only logistic regression (9 raw features)
  3. Random Forest (9 raw features)
  4. XGBoost gradient boosting (9 raw features)
  5. Weather-only logistic regression (T, RH, U only)
  6. FWI baseline logistic regression (Canadian Fire Weather Index)
  7. Physics logistic with SMOTE resampling

Analyses:
  • Ablation study (6 variants)
  • Confusion matrices for all models
  • SHAP values (RF, XGBoost)
  • Threshold analysis (F1 vs threshold)
  • Seasonal performance breakdown
  • Spatial cross-validation (5-fold geographic blocks)
  • Correlation heatmap + feature distributions
  • Log-loss comparison
  • Stratified performance (slope, NDVI)
  • Case study map (Dixie Fire region)

Usage:
    python run_train.py
"""

import os, sys, yaml, warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from src.dataset import build_dataset, split_by_year, get_Xy
from src.models import (
    train_physics_logistic, train_attribute_logistic,
    train_random_forest, train_xgboost, train_weather_logistic,
    save_model, predict_proba,
)
from src.evaluate import (
    compute_metrics, print_metrics, compare_models,
    plot_roc_curves, plot_pr_curves, reliability_diagram,
    performance_by_strata,
)
from src.visualize import (
    plot_feature_distributions, plot_model_comparison_bar,
    plot_R_phys_timeseries, plot_probability_map,
)
from src.analysis import (
    plot_confusion_matrix, plot_shap_values,
    plot_correlation_heatmap, plot_threshold_analysis,
    seasonal_analysis,
)
from src.grid import grid_info


def main():
    with open(os.path.join(ROOT, "config.yaml")) as f:
        cfg = yaml.safe_load(f)

    features_dir = os.path.join(ROOT, cfg["paths"]["features"])
    models_dir   = os.path.join(ROOT, cfg["paths"]["models"])
    outputs_dir  = os.path.join(ROOT, cfg["paths"]["outputs"])
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(outputs_dir, exist_ok=True)

    gi = grid_info(cfg)

    # ═══════════════════════════════════════════════════════════
    # STEP 1: Load & split data
    # ═══════════════════════════════════════════════════════════
    print("=" * 65)
    print("STEP 1 · Loading feature dataset")
    print("=" * 65)

    df = build_dataset(features_dir)
    train_df, val_df, test_df = split_by_year(
        df, cfg["time"]["train_years"],
        cfg["time"]["val_years"], cfg["time"]["test_years"],
    )

    raw_features = ["t_max", "rh_min", "u10_max", "sm_top",
                    "ndvi", "ndwi", "slope", "frp_hist", "count_hist"]
    available_raw = [f for f in raw_features if f in df.columns]
    weather_features = [f for f in ["t_max", "rh_min", "u10_max"] if f in df.columns]

    # ═══════════════════════════════════════════════════════════
    # STEP 2: Correlation heatmap + distributions
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 65)
    print("STEP 2 · Feature analysis")
    print("=" * 65)

    plot_correlation_heatmap(
        train_df, available_raw + ["R_phys", "ignition"],
        save_path=os.path.join(outputs_dir, "correlation_heatmap.png"),
    )
    plot_feature_distributions(
        train_df,
        save_path=os.path.join(outputs_dir, "feature_distributions.png"),
    )

    # ═══════════════════════════════════════════════════════════
    # STEP 3: SMOTE resampling for class imbalance
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 65)
    print("STEP 3 · Class imbalance handling (SMOTE + undersampling)")
    print("=" * 65)

    from imblearn.combine import SMOTETomek
    from imblearn.over_sampling import SMOTE
    from imblearn.under_sampling import RandomUnderSampler
    from imblearn.pipeline import Pipeline as ImbPipeline

    X_train_phys, y_train_phys = get_Xy(train_df, "R_phys")
    X_test_phys, y_test_phys   = get_Xy(test_df, "R_phys")

    X_train_all, y_train_all = get_Xy(train_df, available_raw)
    X_test_all, y_test_all   = get_Xy(test_df, available_raw)
    X_train_all = np.nan_to_num(X_train_all, nan=0.0)
    X_test_all  = np.nan_to_num(X_test_all, nan=0.0)

    X_train_w, y_train_w = get_Xy(train_df, weather_features)
    X_test_w, y_test_w   = get_Xy(test_df, weather_features)
    X_train_w = np.nan_to_num(X_train_w, nan=0.0)
    X_test_w  = np.nan_to_num(X_test_w, nan=0.0)

    # SMOTE + Tomek links combined resampler
    fire_rate = y_train_phys.mean()
    print(f"  Original fire rate: {fire_rate*100:.2f}%   "
          f"({int(y_train_phys.sum()):,} fires / {len(y_train_phys):,} samples)")

    # Use a 1:10 ratio (10% fire) instead of 1:100
    target_ratio = min(0.10, 1.0)
    smote_sampler = ImbPipeline([
        ("smote", SMOTE(sampling_strategy=target_ratio, random_state=42, k_neighbors=3)),
        ("undersampler", RandomUnderSampler(sampling_strategy=0.3, random_state=42)),
    ])

    try:
        X_train_phys_sm, y_train_phys_sm = smote_sampler.fit_resample(
            X_train_phys, y_train_phys.astype(int)
        )
        print(f"  After SMOTE+undersample (R_phys): {len(X_train_phys_sm):,} samples, "
              f"fire rate={y_train_phys_sm.mean()*100:.1f}%")

        X_train_all_sm, y_train_all_sm = smote_sampler.fit_resample(
            X_train_all, y_train_all.astype(int)
        )
        print(f"  After SMOTE+undersample (all):    {len(X_train_all_sm):,} samples, "
              f"fire rate={y_train_all_sm.mean()*100:.1f}%")
        use_smote = True
    except Exception as e:
        print(f"  ⚠ SMOTE failed: {e}, using balanced class weights only")
        X_train_phys_sm, y_train_phys_sm = X_train_phys, y_train_phys
        X_train_all_sm, y_train_all_sm = X_train_all, y_train_all
        use_smote = False

    # ═══════════════════════════════════════════════════════════
    # STEP 4: Train all models
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 65)
    print("STEP 4 · Training models")
    print("=" * 65)

    results = {}
    preds   = {}

    # 1. Physics logistic (balanced weights)
    print("\n─── 1. Physics-Guided Logistic ───")
    m1 = train_physics_logistic(X_train_phys, y_train_phys)
    y_p1 = predict_proba(m1, X_test_phys)
    results["Physics Logistic"] = compute_metrics(y_test_phys, y_p1)
    results["Physics Logistic"]["log_loss"] = float(
        pd.Series(y_test_phys.astype(int)).pipe(
            lambda s: -(s * np.log(np.clip(y_p1, 1e-15, 1)) +
                        (1-s) * np.log(np.clip(1-y_p1, 1e-15, 1))).mean()))
    preds["Physics Logistic"] = (y_test_phys, y_p1)
    print_metrics("Physics Logistic", results["Physics Logistic"])
    save_model(m1, models_dir, "physics_logistic")

    # 2. Physics + SMOTE
    print("\n─── 2. Physics + SMOTE ───")
    m2 = train_physics_logistic(X_train_phys_sm, y_train_phys_sm, C=1.0)
    y_p2 = predict_proba(m2, X_test_phys)
    results["Physics+SMOTE"] = compute_metrics(y_test_phys, y_p2)
    results["Physics+SMOTE"]["log_loss"] = float(
        -(y_test_phys * np.log(np.clip(y_p2, 1e-15, 1)) +
          (1-y_test_phys) * np.log(np.clip(1-y_p2, 1e-15, 1))).mean())
    preds["Physics+SMOTE"] = (y_test_phys, y_p2)
    print_metrics("Physics+SMOTE", results["Physics+SMOTE"])
    save_model(m2, models_dir, "physics_smote")

    # 3. Attribute logistic
    print("\n─── 3. Attribute-Only Logistic ───")
    m3 = train_attribute_logistic(X_train_all, y_train_all)
    y_p3 = predict_proba(m3, X_test_all)
    results["Attribute Logistic"] = compute_metrics(y_test_all, y_p3)
    results["Attribute Logistic"]["log_loss"] = float(
        -(y_test_all * np.log(np.clip(y_p3, 1e-15, 1)) +
          (1-y_test_all) * np.log(np.clip(1-y_p3, 1e-15, 1))).mean())
    preds["Attribute Logistic"] = (y_test_all, y_p3)
    print_metrics("Attribute Logistic", results["Attribute Logistic"])
    save_model(m3, models_dir, "attribute_logistic")

    # 4. Random Forest
    print("\n─── 4. Random Forest ───")
    m4 = train_random_forest(X_train_all_sm, y_train_all_sm)
    y_p4 = predict_proba(m4, X_test_all)
    results["Random Forest"] = compute_metrics(y_test_all, y_p4)
    results["Random Forest"]["log_loss"] = float(
        -(y_test_all * np.log(np.clip(y_p4, 1e-15, 1)) +
          (1-y_test_all) * np.log(np.clip(1-y_p4, 1e-15, 1))).mean())
    preds["Random Forest"] = (y_test_all, y_p4)
    print_metrics("Random Forest", results["Random Forest"])
    save_model(m4, models_dir, "random_forest")

    # 5. XGBoost
    print("\n─── 5. XGBoost ───")
    m5 = train_xgboost(X_train_all_sm, y_train_all_sm)
    y_p5 = predict_proba(m5, X_test_all)
    results["XGBoost"] = compute_metrics(y_test_all, y_p5)
    results["XGBoost"]["log_loss"] = float(
        -(y_test_all * np.log(np.clip(y_p5, 1e-15, 1)) +
          (1-y_test_all) * np.log(np.clip(1-y_p5, 1e-15, 1))).mean())
    preds["XGBoost"] = (y_test_all, y_p5)
    print_metrics("XGBoost", results["XGBoost"])
    save_model(m5, models_dir, "xgboost")

    # 6. Weather-only
    print("\n─── 6. Weather-Only Logistic ───")
    m6 = train_weather_logistic(X_train_w, y_train_w)
    y_p6 = predict_proba(m6, X_test_w)
    results["Weather Only"] = compute_metrics(y_test_w, y_p6)
    results["Weather Only"]["log_loss"] = float(
        -(y_test_w * np.log(np.clip(y_p6, 1e-15, 1)) +
          (1-y_test_w) * np.log(np.clip(1-y_p6, 1e-15, 1))).mean())
    preds["Weather Only"] = (y_test_w, y_p6)
    print_metrics("Weather Only", results["Weather Only"])
    save_model(m6, models_dir, "weather_logistic")

    # 7. FWI baseline
    print("\n─── 7. FWI Baseline ───")
    from src.fwi import compute_fwi_simple, normalize_fwi
    if all(c in test_df.columns for c in ["t_max", "rh_min", "u10_max"]):
        # Compute FWI for train and test
        precip_train = train_df["sm_top"].values if "sm_top" in train_df.columns else np.zeros(len(train_df))
        precip_test  = test_df["sm_top"].values if "sm_top" in test_df.columns else np.zeros(len(test_df))

        # Use unnormalized values (approximate: normed *range + min)
        fwi_train = compute_fwi_simple(
            train_df["t_max"].values * 40, train_df["rh_min"].values * 100,
            train_df["u10_max"].values * 15, precip_train * 10,
        )
        fwi_test = compute_fwi_simple(
            test_df["t_max"].values * 40, test_df["rh_min"].values * 100,
            test_df["u10_max"].values * 15, precip_test * 10,
        )

        fwi_all = np.concatenate([fwi_train, fwi_test])
        fwi_min, fwi_max = fwi_all.min(), fwi_all.max()
        fwi_train_n = (fwi_train - fwi_min) / (fwi_max - fwi_min + 1e-8)
        fwi_test_n  = (fwi_test - fwi_min) / (fwi_max - fwi_min + 1e-8)

        from sklearn.linear_model import LogisticRegression
        m7 = LogisticRegression(solver="lbfgs", max_iter=1000, class_weight="balanced")
        m7.fit(fwi_train_n.reshape(-1, 1), y_train_phys.astype(int))
        y_p7 = m7.predict_proba(fwi_test_n.reshape(-1, 1))[:, 1]
        results["FWI Baseline"] = compute_metrics(y_test_phys, y_p7)
        results["FWI Baseline"]["log_loss"] = float(
            -(y_test_phys * np.log(np.clip(y_p7, 1e-15, 1)) +
              (1-y_test_phys) * np.log(np.clip(1-y_p7, 1e-15, 1))).mean())
        preds["FWI Baseline"] = (y_test_phys, y_p7)
        print_metrics("FWI Baseline", results["FWI Baseline"])
        save_model(m7, models_dir, "fwi_logistic")

    # ═══════════════════════════════════════════════════════════
    # STEP 5: Model comparison (with log-loss)
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 65)
    print("STEP 5 · Model comparison")
    print("=" * 65)

    comp_df = compare_models(results,
        save_path=os.path.join(outputs_dir, "model_comparison.csv"))
    plot_roc_curves(preds, save_path=os.path.join(outputs_dir, "roc_curves.png"))
    plot_pr_curves(preds, save_path=os.path.join(outputs_dir, "pr_curves.png"))
    plot_model_comparison_bar(comp_df,
        save_path=os.path.join(outputs_dir, "model_comparison_bar.png"))

    # ═══════════════════════════════════════════════════════════
    # STEP 6: Confusion matrices
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 65)
    print("STEP 6 · Confusion matrices")
    print("=" * 65)
    for name, (y_t, y_p) in preds.items():
        safe = name.lower().replace(" ", "_").replace("+", "_")
        plot_confusion_matrix(y_t, y_p, model_name=name,
            save_path=os.path.join(outputs_dir, f"confusion_{safe}.png"))

    # ═══════════════════════════════════════════════════════════
    # STEP 7: Threshold + reliability
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 65)
    print("STEP 7 · Threshold analysis + reliability")
    print("=" * 65)
    best_t, best_f1 = plot_threshold_analysis(
        y_test_phys, y_p1, model_name="Physics Logistic",
        save_path=os.path.join(outputs_dir, "threshold_analysis.png"))
    print(f"  Optimal threshold={best_t:.2f}, F1={best_f1:.3f}")
    reliability_diagram(y_test_phys, y_p1,
        save_path=os.path.join(outputs_dir, "reliability_diagram.png"))

    # ═══════════════════════════════════════════════════════════
    # STEP 8: SHAP analysis
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 65)
    print("STEP 8 · SHAP analysis")
    print("=" * 65)
    try:
        plot_shap_values(m4, X_test_all, available_raw, "Random Forest",
            save_path=os.path.join(outputs_dir, "shap_random_forest.png"))
        plot_shap_values(m5, X_test_all, available_raw, "XGBoost",
            save_path=os.path.join(outputs_dir, "shap_xgboost.png"))
    except Exception as e:
        print(f"  ⚠ SHAP skipped: {e}")

    # ═══════════════════════════════════════════════════════════
    # STEP 9: Ablation study
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 65)
    print("STEP 9 · Ablation study")
    print("=" * 65)

    import xarray as xr
    from src.ablation import run_ablation, plot_ablation
    from src.normalize import load_params, normalize

    features_path = os.path.join(features_dir, "features.nc")
    ds = xr.open_dataset(features_path)

    # Build normed data as flat arrays
    normed_flat = {}
    for var in available_raw:
        if var in ds:
            normed_flat[var] = ds[var].values.flatten()

    ignition_flat = ds["ignition"].values.flatten()

    # Build train/test masks for flat arrays
    times = pd.to_datetime(ds.time.values)
    n_lat = len(ds.latitude)
    n_lon = len(ds.longitude)
    n_time = len(times)

    years = np.repeat(times.year.values, n_lat * n_lon)
    train_years = cfg["time"]["train_years"]
    test_years  = cfg["time"]["test_years"]
    train_mask = np.isin(years, train_years)
    test_mask  = np.isin(years, test_years)

    y_abl_train = ignition_flat[train_mask]
    y_abl_test  = ignition_flat[test_mask]

    ablation_results = run_ablation(
        normed_flat, y_abl_train, y_abl_test,
        train_mask, test_mask, cfg,
    )
    abl_df = plot_ablation(ablation_results,
        save_path=os.path.join(outputs_dir, "ablation_study.png"))
    abl_df.to_csv(os.path.join(outputs_dir, "ablation_results.csv"))

    ds.close()

    # ═══════════════════════════════════════════════════════════
    # STEP 10: Spatial cross-validation
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 65)
    print("STEP 10 · Spatial cross-validation")
    print("=" * 65)

    from src.spatial_cv import spatial_kfold

    print("\n  [Physics R_phys – spatial CV]")
    spat_phys, spat_folds = spatial_kfold(
        df, "R_phys", n_folds=5, block_size=2.0)

    if available_raw:
        print("\n  [All features – spatial CV]")
        spat_all, _ = spatial_kfold(
            df, available_raw, n_folds=5, block_size=2.0)

    # ═══════════════════════════════════════════════════════════
    # STEP 11: Seasonal analysis
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 65)
    print("STEP 11 · Seasonal + stratified analysis")
    print("=" * 65)

    if "time" in test_df.columns:
        seasonal_analysis(test_df, y_p1, "Physics Logistic",
            save_path=os.path.join(outputs_dir, "seasonal_analysis.png"))

    if "slope" in test_df.columns:
        performance_by_strata(y_test_phys, y_p1, test_df["slope"].values,
            strata_name="Slope",
            save_path=os.path.join(outputs_dir, "performance_by_slope.csv"))
    if "ndvi" in test_df.columns:
        performance_by_strata(y_test_phys, y_p1, test_df["ndvi"].values,
            strata_name="NDVI",
            save_path=os.path.join(outputs_dir, "performance_by_ndvi.csv"))

    # ═══════════════════════════════════════════════════════════
    # STEP 12: R_phys timeseries
    # ═══════════════════════════════════════════════════════════
    if "time" in test_df.columns:
        ts = test_df.groupby("time").agg(
            R_phys_mean=("R_phys", "mean"),
            fire_rate=("ignition", "mean"),
        ).reset_index()
        plot_R_phys_timeseries(ts["time"], ts["R_phys_mean"], ts["fire_rate"],
            save_path=os.path.join(outputs_dir, "R_phys_timeseries.png"))

    # ═══════════════════════════════════════════════════════════
    # STEP 13: Case study map (Dixie Fire region)
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 65)
    print("STEP 13 · Case study map (Dixie Fire region)")
    print("=" * 65)

    # Dixie Fire: July 2021, ~40.0°N, -121.4°W
    dixie_lat, dixie_lon = 40.0, -121.4
    dixie_date = "2021-07-15"

    ds2 = xr.open_dataset(features_path)
    t_idx = np.argmin(np.abs(ds2.time.values - np.datetime64(dixie_date)))
    if "R_phys" in ds2:
        R_day = ds2["R_phys"].isel(time=t_idx).values
        a_coef = m1.coef_[0, 0]
        b_coef = m1.intercept_[0]
        prob_day = 1 / (1 + np.exp(-(a_coef * R_day + b_coef)))

        plot_probability_map(
            prob_day, gi["lats"], gi["lons"],
            date_str=dixie_date,
            title=f"Case Study: Dixie Fire Region — {dixie_date}\n"
                  f"★ Dixie Fire origin: ({dixie_lat}°N, {dixie_lon}°W)",
            save_path=os.path.join(outputs_dir, "case_study_dixie_fire.png"),
        )

        # Find probability at fire location
        lat_idx = np.argmin(np.abs(gi["lats"] - dixie_lat))
        lon_idx = np.argmin(np.abs(gi["lons"] - dixie_lon))
        p_fire = prob_day[lat_idx, lon_idx]
        print(f"  Model probability at Dixie Fire origin: {p_fire:.4f}")
    ds2.close()

    # ═══════════════════════════════════════════════════════════
    # SUMMARY
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 65)
    print("✅ TRAINING & EVALUATION COMPLETE")
    print("=" * 65)
    print(f"   Models:  {models_dir}/")
    print(f"   Outputs: {outputs_dir}/")
    print(f"\n   {len(results)} models trained:")
    for name in results:
        print(f"     • {name}")
    print(f"\n   Generated files:")
    for f in sorted(os.listdir(outputs_dir)):
        if f.endswith((".png", ".csv")):
            print(f"     • {f}")


if __name__ == "__main__":
    main()
