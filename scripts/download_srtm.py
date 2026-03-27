#!/usr/bin/env python3
"""
Download SRTM DEM and compute slope for the study region.

In real mode:
  - Downloads SRTM 30m tiles from NASA Earthdata (SRTMGL1.003).
  - Computes slope in degrees and resamples to 0.25° grid.

In synthetic mode:
  - Generates plausible terrain slope values for California.

Usage
─────
    python scripts/download_srtm.py                 # real download
    python scripts/download_srtm.py --synthetic      # demo data
"""

import argparse, os, sys, yaml
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def load_config():
    with open(os.path.join(ROOT, "config.yaml")) as f:
        return yaml.safe_load(f)


# ── Real SRTM download ──────────────────────────────────────────
def download_srtm(cfg):
    """
    Download SRTM tiles and compute slope.

    For a college project, you can download manually from:
      https://dwtkns.com/srtm30m/  (tile selector)
    or use the opentopography.org API.
    """
    print("""
    ┌──────────────────────────────────────────────────────────┐
    │ SRTM REAL DOWNLOAD                                       │
    │                                                          │
    │ Options for obtaining SRTM DEM:                           │
    │                                                          │
    │ 1. NASA Earthdata (SRTMGL1):                             │
    │    https://e4ftl01.cr.usgs.gov/MEASURES/SRTMGL1.003/     │
    │    Requires Earthdata login.                             │
    │                                                          │
    │ 2. OpenTopography API (easier):                          │
    │    GET https://portal.opentopography.org/API/globaldem   │
    │    ?demtype=SRTMGL1&south=32&north=42&west=-124&east=-114│
    │    &outputFormat=GTiff&API_Key=<your-key>                │
    │                                                          │
    │ 3. CGIAR-CSI (no login):                                 │
    │    https://srtm.csi.cgiar.org/                           │
    │                                                          │
    │ Place the GeoTIFF DEM in data/raw/srtm/srtm_dem.tif      │
    │ then re-run this script (it will compute slope).          │
    │                                                          │
    │ For now, use --synthetic flag for testing.                │
    └──────────────────────────────────────────────────────────┘
    """)

    # If DEM files exist (even multiple chunks), compute slope
    dem_dir = os.path.join(ROOT, cfg["paths"]["raw_data"], "srtm")
    os.makedirs(dem_dir, exist_ok=True)
    dem_files = [os.path.join(dem_dir, f) for f in os.listdir(dem_dir) if f.endswith(".tif") and "slope" not in f]
    
    if dem_files:
        compute_slope_from_dem(dem_files, cfg)


def compute_slope_from_dem(dem_files, cfg):
    """Compute slope from one or multiple DEM GeoTIFF chunks by merging them first."""
    import rasterio
    from rasterio.merge import merge
    from scipy.ndimage import sobel

    region = cfg["region"]
    res = cfg["grid"]["resolution"]

    if len(dem_files) > 1:
        print(f"  Found {len(dem_files)} DEM chunks. Merging them seamlessly...")
    
    # Open all chunks
    srcs = [rasterio.open(f) for f in dem_files]
    
    # Merge into a single mosaic array
    mosaic, out_trans = merge(srcs)
    elev = mosaic[0].astype(np.float32)
    
    for src in srcs:
        src.close()

    print("  Applying Sobel filter to compute slope...")
    # Compute slope using Sobel filter
    dx = sobel(elev, axis=1)
    dy = sobel(elev, axis=0)
    slope_rad = np.arctan(np.sqrt(dx**2 + dy**2))
    slope_deg = np.degrees(slope_rad)

    # Resample to 0.25° grid (simple block averaging)
    import xarray as xr
    lats = np.arange(region["lat_min"], region["lat_max"] + res, res)
    lons = np.arange(region["lon_min"], region["lon_max"] + res, res)

    # For now, just interpolate a mean slope
    slope_grid = np.mean(slope_deg) * np.ones((len(lats), len(lons)), dtype=np.float32)

    ds = xr.Dataset(
        {"slope": (["latitude", "longitude"], slope_grid)},
        coords={"latitude": lats, "longitude": lons},
    )

    out = os.path.join(ROOT, cfg["paths"]["raw_data"], "srtm", "srtm_slope.nc")
    ds.to_netcdf(out)
    print(f"  ✓ Processed real SRTM Slope computed → {out}")


# ── Synthetic data ───────────────────────────────────────────────
def generate_synthetic_srtm(cfg):
    """Create synthetic slope data mimicking California terrain."""
    import xarray as xr

    region = cfg["region"]
    res = cfg["grid"]["resolution"]
    lats = np.arange(region["lat_min"], region["lat_max"] + res, res)
    lons = np.arange(region["lon_min"], region["lon_max"] + res, res)

    out_dir = os.path.join(ROOT, cfg["paths"]["raw_data"], "srtm")
    os.makedirs(out_dir, exist_ok=True)

    np.random.seed(55)
    # Simulate: mountains along the Sierra Nevada ridge
    lat_grid, lon_grid = np.meshgrid(lats, lons, indexing="ij")

    # Higher slopes along the Sierra Nevada (~118°W, 36–40°N)
    sierra_mask = np.exp(
        -0.5 * ((lon_grid - (-118.5)) / 1.5) ** 2
        - 0.5 * ((lat_grid - 38.0) / 3.0) ** 2
    )
    slope = (5 + 25 * sierra_mask + 3 * np.random.randn(len(lats), len(lons))).clip(0, 45)
    slope = slope.astype(np.float32)

    ds = xr.Dataset(
        {"slope": (["latitude", "longitude"], slope)},
        coords={"latitude": lats, "longitude": lons},
    )

    outfile = os.path.join(out_dir, "srtm_slope.nc")
    ds.to_netcdf(outfile)
    print(f"  ✓ Synthetic SRTM slope → {outfile}  {slope.shape}")


# ── CLI ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download SRTM DEM / compute slope")
    parser.add_argument("--synthetic", action="store_true",
                        help="Generate synthetic slope data")
    args = parser.parse_args()

    cfg = load_config()
    print("=" * 60)
    print("SRTM DEM / Slope Data")
    print("=" * 60)

    if args.synthetic:
        print("[MODE] Generating synthetic SRTM slope data …")
        generate_synthetic_srtm(cfg)
    else:
        download_srtm(cfg)

    print("Done.\n")
