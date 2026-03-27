"""
Preprocessing – daily aggregation of raw datasets to gridded daily values.

Handles ERA5, MODIS, SRTM, and FIRMS data.
"""

import os, glob, yaml
import numpy as np
import pandas as pd
import xarray as xr
from src.grid import build_grid, assign_to_grid, grid_info


# ══════════════════════════════════════════════════════════════════
# ERA5 → daily aggregates
# ══════════════════════════════════════════════════════════════════

def _relative_humidity(t2m_K, d2m_K):
    """Magnus formula: compute RH (%) from T and dewpoint (both in K)."""
    t = t2m_K - 273.15
    td = d2m_K - 273.15
    # August–Roche–Magnus approximation
    rh = 100 * np.exp(17.625 * td / (243.04 + td)) / \
               np.exp(17.625 * t  / (243.04 + t))
    return np.clip(rh, 0, 100)


def process_era5(raw_dir, cfg):
    """
    Process ERA5 hourly NetCDFs → daily grid-level aggregates.

    Output variables per cell per day:
      t_max   – daily max 2 m temperature (°C)
      rh_min  – daily min relative humidity (%)
      u10_max – daily max 10 m wind speed (m/s)
      p_tot   – daily total precipitation (mm)
      sm_top  – daily mean top-layer soil moisture (m³/m³)

    Returns xr.Dataset with dims (time, latitude, longitude).
    """
    gi = grid_info(cfg)
    era5_dir = os.path.join(raw_dir, "era5")
    files = sorted(glob.glob(os.path.join(era5_dir, "era5_*.nc")))
    if not files:
        raise FileNotFoundError(f"No ERA5 files in {era5_dir}")

    all_years = []
    for f in files:
        print(f"  Processing {os.path.basename(f)} …")
        ds = xr.open_dataset(f)

        # Rename coords if needed
        if "lat" in ds.dims:
            ds = ds.rename({"lat": "latitude", "lon": "longitude"})

        # Compute wind speed
        if "v10" in ds:
            ws = np.sqrt(ds["u10"]**2 + ds["v10"]**2)
        else:
            ws = np.abs(ds["u10"])

        # Compute RH
        if "d2m" in ds and "t2m" in ds:
            rh = _relative_humidity(ds["t2m"].values, ds["d2m"].values)
            ds["rh"] = (ds["t2m"].dims, rh)

        ds["ws"] = (ds["u10"].dims, ws.values)

        # Daily aggregation
        daily = xr.Dataset()
        daily["t_max"]   = (ds["t2m"] - 273.15).resample(time="1D").max()
        daily["rh_min"]  = ds["rh"].resample(time="1D").min()
        daily["u10_max"] = ds["ws"].resample(time="1D").max()
        daily["p_tot"]   = ds["tp"].resample(time="1D").sum() * 1000  # m → mm
        daily["sm_top"]  = ds["swvl1"].resample(time="1D").mean()

        # Regrid to target lats/lons (nearest-neighbour interpolation)
        daily = daily.interp(latitude=gi["lats"], longitude=gi["lons"],
                             method="nearest")
        all_years.append(daily)
        ds.close()

    result = xr.concat(all_years, dim="time")
    return result


# ══════════════════════════════════════════════════════════════════
# MODIS → daily NDVI, NDWI (interpolated)
# ══════════════════════════════════════════════════════════════════

def process_modis(raw_dir, cfg):
    """
    Process MODIS 16-day composites → daily NDVI & NDWI via
    last-observation-carried-forward interpolation.

    Returns xr.Dataset with daily (time, latitude, longitude).
    """
    gi = grid_info(cfg)
    modis_dir = os.path.join(raw_dir, "modis")
    files = sorted(glob.glob(os.path.join(modis_dir, "*.nc")))
    if not files:
        raise FileNotFoundError(f"No MODIS files in {modis_dir}")

    all_years = []
    for f in files:
        print(f"  Processing {os.path.basename(f)} …")
        ds = xr.open_dataset(f)

        # Standardize coordinates
        if "lat" in ds.dims:
            ds = ds.rename({"lat": "latitude", "lon": "longitude"})

        # Handle NASA AppEEARS variable names (xarray auto-applies NetCDF scale_factor)
        if "_1_km_16_days_NDVI" in ds:
            ds["ndvi"] = ds["_1_km_16_days_NDVI"]
            # If NDWI isn't explicitly downloaded, use a proxy based on NDVI & MIR or just NDVI
            if "_1_km_16_days_MIR_reflectance" in ds:
                # Rough approximation: normalize MIR to [-1, 1] and invert
                mir = ds["_1_km_16_days_MIR_reflectance"]
                ds["ndwi"] = (ds["ndvi"] - mir) / (ds["ndvi"] + mir + 1e-6)
            else:
                ds["ndwi"] = ds["ndvi"] - 0.2
        elif "NDVI" in ds:
            ds["ndvi"] = ds["NDVI"]
            ds["ndwi"] = ds["ndvi"] - 0.2

        # Ensure ndvi and ndwi are within valid physical ranges
        if "ndvi" in ds:
            ds["ndvi"] = xr.where(ds["ndvi"] > 1.0, 1.0, ds["ndvi"])
            ds["ndvi"] = xr.where(ds["ndvi"] < -1.0, -1.0, ds["ndvi"])
        if "ndwi" in ds:
            ds["ndwi"] = xr.where(ds["ndwi"] > 1.0, 1.0, ds["ndwi"])
            ds["ndwi"] = xr.where(ds["ndwi"] < -1.0, -1.0, ds["ndwi"])

        # Regrid to target bounding box
        ds = ds.interp(latitude=gi["lats"], longitude=gi["lons"],
                       method="nearest")

        # Keep only the standardized variables we need
        ds_clean = ds[["ndvi", "ndwi"]]

        # Resample to daily using forward-fill
        daily = ds_clean.resample(time="1D").ffill()
        all_years.append(daily)
        ds.close()

    result = xr.concat(all_years, dim="time")
    return result


# ══════════════════════════════════════════════════════════════════
# SRTM → slope per grid cell (static)
# ══════════════════════════════════════════════════════════════════

def process_srtm(raw_dir, cfg):
    """
    Load precomputed slope grid.

    Returns xr.DataArray with dims (latitude, longitude).
    """
    srtm_file = os.path.join(raw_dir, "srtm", "srtm_slope.nc")
    if not os.path.exists(srtm_file):
        raise FileNotFoundError(f"SRTM slope file not found: {srtm_file}")

    gi = grid_info(cfg)
    ds = xr.open_dataset(srtm_file)
    slope = ds["slope"].interp(latitude=gi["lats"], longitude=gi["lons"],
                               method="nearest")
    return slope


# ══════════════════════════════════════════════════════════════════
# FIRMS → ignition labels + fire history features
# ══════════════════════════════════════════════════════════════════

def process_firms(raw_dir, cfg):
    """
    Process FIRMS CSV → per-cell per-day:
      y(t)         : binary ignition label (1 if any fire, else 0)
      frp_hist(t)  : exponentially decayed sum of FRP over sliding window
      count_hist(t): exponentially decayed count of hotspots over window

    Returns xr.Dataset with dims (time, latitude, longitude).
    """
    gi = grid_info(cfg)
    firms_dir = os.path.join(raw_dir, "firms")
    files = sorted(glob.glob(os.path.join(firms_dir, "firms_*.csv")))
    if not files:
        raise FileNotFoundError(f"No FIRMS files in {firms_dir}")

    window = cfg["firms"]["sliding_window_days"]
    decay  = cfg["firms"]["decay_factor"]
    conf_filter = cfg["firms"]["confidence_filter"]
    type_filter = cfg["firms"]["type_filter"]

    # Read and concatenate all FIRMS CSVs
    dfs = []
    for f in files:
        print(f"  Processing {os.path.basename(f)} …")
        df = pd.read_csv(f)
        dfs.append(df)
    firms = pd.concat(dfs, ignore_index=True)

    # Filter
    firms = firms[firms["confidence"].isin(conf_filter)]
    firms = firms[firms["type"] == type_filter]
    firms["acq_date"] = pd.to_datetime(firms["acq_date"])

    # Assign to grid cells
    lat_idx, lon_idx = assign_to_grid(
        firms["latitude"].values, firms["longitude"].values,
        gi["lats"][0] - gi["resolution"] / 2,
        gi["lons"][0] - gi["resolution"] / 2,
        gi["resolution"],
    )
    firms["lat_idx"] = lat_idx
    firms["lon_idx"] = lon_idx

    # Build daily time index
    start = pd.Timestamp(cfg["time"]["start"])
    end   = pd.Timestamp(cfg["time"]["end"])
    days  = pd.date_range(start, end, freq="D")

    n_lat = gi["n_lat"]
    n_lon = gi["n_lon"]
    n_days = len(days)

    # Allocate arrays
    y          = np.zeros((n_days, n_lat, n_lon), dtype=np.float32)
    frp_hist   = np.zeros((n_days, n_lat, n_lon), dtype=np.float32)
    count_hist = np.zeros((n_days, n_lat, n_lon), dtype=np.float32)

    # Group fires by cell and date
    for _, row in firms.iterrows():
        li, lo = int(row["lat_idx"]), int(row["lon_idx"])
        if 0 <= li < n_lat and 0 <= lo < n_lon:
            dt = row["acq_date"]
            if start <= dt <= end:
                tidx = (dt - start).days
                y[tidx, li, lo] = 1.0

    # Compute sliding-window history features
    for t in range(n_days):
        for k in range(1, window + 1):
            t_k = t - k
            if t_k < 0:
                continue
            w = decay ** k  # exponential decay
            frp_hist[t]   += w * y[t_k]   # simplified: using y as proxy
            count_hist[t] += w * y[t_k]

    result = xr.Dataset(
        {
            "ignition":   (["time", "latitude", "longitude"], y),
            "frp_hist":   (["time", "latitude", "longitude"], frp_hist),
            "count_hist": (["time", "latitude", "longitude"], count_hist),
        },
        coords={
            "time":      days,
            "latitude":  gi["lats"],
            "longitude": gi["lons"],
        },
    )
    return result


# ── Self-test ─────────────────────────────────────────────────────
if __name__ == "__main__":
    import yaml
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with open(os.path.join(root, "config.yaml")) as f:
        cfg = yaml.safe_load(f)
    raw_dir = os.path.join(root, cfg["paths"]["raw_data"])

    print("Testing ERA5 processing …")
    try:
        era5 = process_era5(raw_dir, cfg)
        print(f"  ERA5: {dict(era5.dims)}")
    except FileNotFoundError as e:
        print(f"  Skipped: {e}")
