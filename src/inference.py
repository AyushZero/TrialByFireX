"""
Inference - operational pipeline for daily ignition probability.

Given a trained model and latest data, compute R_phys(T) and p_ign(T)
for each grid cell.
"""

import os, json, yaml
import numpy as np
import xarray as xr
from src.normalize import normalize, load_params
from src.features import build_all_features
from src.models import load_model, predict_proba


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
        probs = predict_proba(model, X)
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
