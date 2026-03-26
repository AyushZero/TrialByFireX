"""
Grid builder – creates a regular 0.25° lat/lon grid and provides
utilities to assign point data (lat/lon) to grid cells.
"""

import numpy as np
import xarray as xr


def build_grid(lat_min, lat_max, lon_min, lon_max, resolution=0.25):
    """
    Build a regular lat/lon grid.

    Returns
    -------
    lats : ndarray – cell-centre latitudes
    lons : ndarray – cell-centre longitudes
    """
    lats = np.arange(lat_min + resolution / 2, lat_max, resolution)
    lons = np.arange(lon_min + resolution / 2, lon_max, resolution)
    return lats, lons


def assign_to_grid(lat, lon, lat_min, lon_min, resolution=0.25):
    """
    Map point coordinates to grid cell indices.

    Parameters
    ----------
    lat, lon : float or array-like
    lat_min, lon_min : float – grid origin
    resolution : float

    Returns
    -------
    lat_idx, lon_idx : int or ndarray
    """
    lat = np.asarray(lat)
    lon = np.asarray(lon)
    lat_idx = ((lat - lat_min) / resolution).astype(int)
    lon_idx = ((lon - lon_min) / resolution).astype(int)
    return lat_idx, lon_idx


def grid_info(cfg):
    """Return grid parameters from config dict."""
    r = cfg["region"]
    res = cfg["grid"]["resolution"]
    lats, lons = build_grid(r["lat_min"], r["lat_max"],
                            r["lon_min"], r["lon_max"], res)
    return {
        "lats": lats,
        "lons": lons,
        "resolution": res,
        "lat_min": r["lat_min"],
        "lon_min": r["lon_min"],
        "n_lat": len(lats),
        "n_lon": len(lons),
    }


# ── Self-test ─────────────────────────────────────────────────────
if __name__ == "__main__":
    lats, lons = build_grid(32, 42, -124, -114, 0.25)
    print(f"Grid: {len(lats)} lat × {len(lons)} lon = {len(lats)*len(lons)} cells")
    print(f"Lat range: {lats[0]:.2f} – {lats[-1]:.2f}")
    print(f"Lon range: {lons[0]:.2f} – {lons[-1]:.2f}")

    # Test point assignment
    li, lo = assign_to_grid(36.5, -119.2, 32, -124, 0.25)
    print(f"Point (36.5, -119.2) → cell ({li}, {lo})")
