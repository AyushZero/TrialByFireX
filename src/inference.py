"""
Inference - operational pipeline for daily ignition probability.

Given a trained model and latest data, compute R_phys(T) and p_ign(T)
for each grid cell.
"""

import os, json, yaml
import numpy as np
import xarray as xr
import pandas as pd
from src.normalize import normalize, load_params
from src.features import build_all_features
from src.models import load_model, predict_proba


def _safe_predict_proba(model, X):
    """
    Predict probabilities while handling missing/non-finite rows.

    Returns
    -------
    probs : np.ndarray, shape (n_samples,)
        Probabilities for valid rows and NaN for invalid rows.
    """
    X = np.asarray(X)
    if X.ndim == 1:
        X = X.reshape(-1, 1)

    finite_mask = np.isfinite(X).all(axis=1)
    probs = np.full(X.shape[0], np.nan, dtype=float)

    if finite_mask.any():
        probs[finite_mask] = predict_proba(model, X[finite_mask])
    else:
        raise ValueError("No valid finite feature rows available for inference")

    n_bad = int((~finite_mask).sum())
    if n_bad > 0:
        print(f"  [WARN] Skipped {n_bad} cells with non-finite features during inference")

    return probs


def run_inference(date_str, cfg, model_dir, norm_params_path, output_dir):
    """
    Run daily inference for a given date.

    Steps:
    1. Load latest data (ERA5 + MODIS + FIRMS history up to date-1)
    2. Normalise using stored parameters
    3. Compute physics features -> R_phys
    4. Apply logistic model -> p_ign
    5. Write output probability raster

    Parameters
    ----------
    date_str : str - "YYYY-MM-DD"
    cfg : dict - config
    model_dir : str - path to saved model
    norm_params_path : str - path to norm params JSON
    output_dir : str - where to write output files

    Returns
    -------
    prob_grid : xr.DataArray - probability of ignition per cell
    """
    from src.grid import grid_info

    gi = grid_info(cfg)
    n_lat = gi["n_lat"]
    n_lon = gi["n_lon"]

    # Load model
    model = load_model(model_dir, "physics_logistic")
    norm_params = load_params(norm_params_path)

    # In a real pipeline, load actual data here.
    # For demonstration, we generate placeholder data:
    print(f"  Running inference for {date_str} ...")
    print(f"  Grid: {n_lat} x {n_lon} cells")

    # Placeholder: load from features directory
    features_dir = os.path.join(os.path.dirname(model_dir), "data", "features")
    features_path = os.path.join(features_dir, "features.nc")

    if os.path.exists(features_path):
        ds = xr.open_dataset(features_path)
        # Find the closest date
        target = np.datetime64(date_str)
        time_idx = np.argmin(np.abs(ds.time.values - target))
        slice_ds = ds.isel(time=time_idx)

        R_phys = slice_ds["R_phys"].values.flatten()
        X = R_phys.reshape(-1, 1)
        probs = _safe_predict_proba(model, X)
        prob_grid = probs.reshape(n_lat, n_lon)
    else:
        # Demo: random probabilities
        np.random.seed(hash(date_str) % 2**32)
        prob_grid = np.random.rand(n_lat, n_lon) * 0.3

    # Build output DataArray
    result = xr.DataArray(
        prob_grid,
        dims=["latitude", "longitude"],
        coords={"latitude": gi["lats"], "longitude": gi["lons"]},
        name="ignition_probability",
        attrs={
            "units": "probability",
            "description": f"Ignition probability for {date_str}",
            "model": "physics-guided logistic regression",
        },
    )

    # Save as NetCDF
    os.makedirs(output_dir, exist_ok=True)
    outpath = os.path.join(output_dir, f"p_ign_{date_str}.nc")
    result.to_netcdf(outpath)
    print(f"  [OK] Probability map -> {outpath}")

    return result


def write_geotiff(prob_da, output_path):
    """
    Write an xarray DataArray to GeoTIFF.

    Requires rasterio.
    """
    import rasterio
    from rasterio.transform import from_bounds

    data = prob_da.values
    lats = prob_da.latitude.values
    lons = prob_da.longitude.values

    transform = from_bounds(
        lons.min(), lats.min(), lons.max(), lats.max(),
        len(lons), len(lats),
    )

    with rasterio.open(
        output_path, "w",
        driver="GTiff",
        height=data.shape[0],
        width=data.shape[1],
        count=1,
        dtype=data.dtype,
        crs="EPSG:4326",
        transform=transform,
    ) as dst:
        dst.write(data, 1)

    print(f"  [OK] GeoTIFF -> {output_path}")


def run_inference_from_daily_dataframe(
    day_df,
    cfg,
    model_dir,
    norm_params_path,
    normalized_input=True,
    model_name="physics_logistic",
):
    """
    Run single-day inference from a provided daily grid dataframe.

    Expected columns:
      latitude, longitude, and either:
      - R_phys
      - or raw/normalized feature columns
        [t_max, rh_min, u10_max, sm_top, ndvi, ndwi, slope, frp_hist, count_hist]
    """
    from src.grid import grid_info, assign_to_grid
    from src.features import (
        compute_F_avail,
        compute_F_dry,
        compute_G_spread,
        compute_H_history,
        compute_R_phys,
    )

    if not isinstance(day_df, pd.DataFrame):
        day_df = pd.DataFrame(day_df)

    required_loc = ["latitude", "longitude"]
    for col in required_loc:
        if col not in day_df.columns:
            raise ValueError(f"Missing required column '{col}'")

    gi = grid_info(cfg)
    model = load_model(model_dir, model_name)
    norm_params = load_params(norm_params_path)

    work = day_df.copy()

    feature_cols = [
        "t_max", "rh_min", "u10_max", "sm_top", "ndvi",
        "ndwi", "slope", "frp_hist", "count_hist",
    ]

    if "R_phys" in work.columns:
        r_vals = work["R_phys"].to_numpy(dtype=float)
    else:
        missing = [c for c in feature_cols if c not in work.columns]
        if missing:
            raise ValueError(
                "Missing feature columns for inference: " + ", ".join(missing)
            )

        normed = {}
        for c in feature_cols:
            vals = work[c].to_numpy(dtype=float)
            if normalized_input:
                normed[c] = np.clip(vals, 0.0, 1.0)
            else:
                normed[c] = normalize(vals, norm_params[c])

        f_avail = compute_F_avail(normed["ndvi"])
        f_dry = compute_F_dry(normed["ndwi"], normed["rh_min"], normed["sm_top"], cfg["alpha"])
        g_spread = compute_G_spread(normed["u10_max"], normed["slope"], cfg["beta"])
        h_history = compute_H_history(normed["frp_hist"], normed["count_hist"], cfg["gamma"])
        r_vals = compute_R_phys(normed["t_max"], f_avail, f_dry, g_spread, h_history)

    lat_idx, lon_idx = assign_to_grid(
        work["latitude"].to_numpy(dtype=float),
        work["longitude"].to_numpy(dtype=float),
        gi["lat_min"],
        gi["lon_min"],
        gi["resolution"],
    )

    valid = (
        np.isfinite(r_vals)
        & np.isfinite(lat_idx)
        & np.isfinite(lon_idx)
        & (lat_idx >= 0)
        & (lon_idx >= 0)
        & (lat_idx < gi["n_lat"])
        & (lon_idx < gi["n_lon"])
    )

    prob_grid = np.full((gi["n_lat"], gi["n_lon"]), np.nan, dtype=float)
    if valid.any():
        x = np.asarray(r_vals[valid], dtype=float).reshape(-1, 1)
        p = _safe_predict_proba(model, x)
        prob_grid[lat_idx[valid], lon_idx[valid]] = p

    result = xr.DataArray(
        prob_grid,
        dims=["latitude", "longitude"],
        coords={"latitude": gi["lats"], "longitude": gi["lons"]},
        name="ignition_probability",
        attrs={
            "units": "probability",
            "description": "Ignition probability from provided daily dataframe",
            "model": f"physics-guided logistic regression ({model_name})",
        },
    )

    meta = {
        "rows_total": int(len(work)),
        "rows_valid": int(valid.sum()),
        "rows_dropped": int((~valid).sum()),
    }
    return result, meta
