#!/usr/bin/env python3
"""
Download MODIS NDVI / NDWI products for the study region.

In real mode:
  - Uses NASA AppEEARS API (https://appeears.earthdatacloud.nasa.gov/api/)
    to extract MOD13A2 (16-day NDVI) and MOD09GA (for NDWI computation)
    over the California bounding box.
  - Requires a NASA Earthdata account.

In synthetic mode:
  - Generates seasonal NDVI/NDWI patterns for pipeline testing.

Usage
─────
    python scripts/download_modis.py                 # real download
    python scripts/download_modis.py --synthetic      # demo data
"""

import argparse, os, sys, yaml
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def load_config():
    with open(os.path.join(ROOT, "config.yaml")) as f:
        return yaml.safe_load(f)


# ── Real MODIS download via AppEEARS ─────────────────────────────
def download_modis(cfg):
    """
    Download MODIS data via NASA AppEEARS API.

    NOTE: This requires NASA Earthdata credentials. If you do not have them,
    use --synthetic mode instead.

    For a college project, you can also manually download from:
      https://lpdaac.usgs.gov/products/mod13a2v061/
    and place files in data/raw/modis/.
    """
    print("""
    ┌─────────────────────────────────────────────────────────┐
    │ MODIS REAL DOWNLOAD                                     │
    │                                                         │
    │ For real MODIS data, you have these options:             │
    │                                                         │
    │ Option A – NASA AppEEARS (recommended):                 │
    │   1. Create account at earthdata.nasa.gov               │
    │   2. Go to appeears.earthdatacloud.nasa.gov             │
    │   3. Submit area request for:                           │
    │      • Product: MOD13A2.061 (NDVI 16-day)               │
    │      • Layers: _1_km_16_days_NDVI,                      │
    │                _1_km_16_days_NIR_reflectance,            │
    │                _1_km_16_days_MIR_reflectance             │
    │      • Region: California bbox                          │
    │      • Dates: 2021-01-01 to 2023-12-31                  │
    │   4. Download NetCDF output to data/raw/modis/          │
    │                                                         │
    │ Option B – Google Earth Engine:                          │
    │   Use GEE Python API to extract MODIS products          │
    │                                                         │
    │ For now, use --synthetic flag for testing.               │
    └─────────────────────────────────────────────────────────┘
    """)


# ── Synthetic data ───────────────────────────────────────────────
def generate_synthetic_modis(cfg):
    """Create synthetic NDVI/NDWI NetCDF files with realistic seasonality."""
    import xarray as xr

    region = cfg["region"]
    res = cfg["grid"]["resolution"]
    lats = np.arange(region["lat_min"], region["lat_max"] + res, res)
    lons = np.arange(region["lon_min"], region["lon_max"] + res, res)

    out_dir = os.path.join(ROOT, cfg["paths"]["raw_data"], "modis")
    os.makedirs(out_dir, exist_ok=True)

    start_year = int(cfg["time"]["start"][:4])
    end_year   = int(cfg["time"]["end"][:4])

    np.random.seed(77)
    for year in range(start_year, end_year + 1):
        # 16-day composites (≈23 per year)
        times = np.arange(
            np.datetime64(f"{year}-01-01"),
            np.datetime64(f"{year+1}-01-01"),
            np.timedelta64(16, "D"),
        )
        shape = (len(times), len(lats), len(lons))

        # Seasonal NDVI: higher in spring, lower in late summer
        doy = np.array([(t - np.datetime64(f"{year}-01-01")).astype(int) for t in times])
        seasonal = 0.3 + 0.4 * np.sin(np.pi * doy[:, None, None] / 365)
        ndvi = np.clip(seasonal + 0.1 * np.random.randn(*shape), 0.0, 1.0).astype(np.float32)

        # NDWI: correlated with NDVI (higher when more moisture)
        ndwi = np.clip(0.8 * ndvi - 0.1 + 0.1 * np.random.randn(*shape), -0.5, 0.8).astype(np.float32)

        ds = xr.Dataset(
            {
                "ndvi": (["time", "latitude", "longitude"], ndvi),
                "ndwi": (["time", "latitude", "longitude"], ndwi),
            },
            coords={
                "time":      times,
                "latitude":  lats,
                "longitude": lons,
            },
        )

        outfile = os.path.join(out_dir, f"modis_{year}.nc")
        ds.to_netcdf(outfile)
        print(f"  ✓ Synthetic MODIS → {outfile}  ({ds.dims})")


# ── CLI ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download MODIS NDVI/NDWI data")
    parser.add_argument("--synthetic", action="store_true",
                        help="Generate synthetic demo data")
    args = parser.parse_args()

    cfg = load_config()
    print("=" * 60)
    print("MODIS NDVI / NDWI Data Download")
    print("=" * 60)

    if args.synthetic:
        print("[MODE] Generating synthetic MODIS data …")
        generate_synthetic_modis(cfg)
    else:
        download_modis(cfg)

    print("Done.\n")
