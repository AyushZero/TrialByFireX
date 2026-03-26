#!/usr/bin/env python3
"""
Download ERA5 reanalysis data for the study region via the CDS API.

Prerequisites
─────────────
1. pip install cdsapi
2. Create a file  ~/.cdsapirc  (or %USERPROFILE%\.cdsapirc on Windows) with:

       url: https://cds.climate.copernicus.eu/api
       key: <your-key>

   Alternatively, the script reads from config.yaml → era5 section.

Usage
─────
    python scripts/download_era5.py                 # real download
    python scripts/download_era5.py --synthetic      # generate demo data
"""

import argparse, os, sys, yaml
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def load_config():
    with open(os.path.join(ROOT, "config.yaml")) as f:
        return yaml.safe_load(f)


# ── Real ERA5 download via CDS API ────────────────────────────────
def download_era5(cfg):
    """Download ERA5 hourly data year-by-year as NetCDF."""
    import cdsapi

    region = cfg["region"]
    area = [region["lat_max"], region["lon_min"],
            region["lat_min"], region["lon_max"]]  # N, W, S, E

    out_dir = os.path.join(ROOT, cfg["paths"]["raw_data"], "era5")
    os.makedirs(out_dir, exist_ok=True)

    start_year = int(cfg["time"]["start"][:4])
    end_year   = int(cfg["time"]["end"][:4])

    variables = cfg["era5"]["variables"]

    client = cdsapi.Client()

    for year in range(start_year, end_year + 1):
        print(f"  Requesting ERA5 for {year} (chunked by month) ...")
        for m in range(1, 13):
            month_str = f"{m:02d}"
            outfile = os.path.join(out_dir, f"era5_{year}_{month_str}.nc")
            if os.path.exists(outfile):
                print(f"  [skip] {outfile} already exists")
                continue

            days   = [f"{d:02d}" for d in range(1, 32)]
            hours  = [f"{h:02d}:00" for h in range(0, 24)]

            try:
                client.retrieve(
                    "reanalysis-era5-single-levels",
                    {
                        "product_type": "reanalysis",
                        "variable": variables,
                        "year":  str(year),
                        "month": month_str,
                        "day":   days,
                        "time":  hours,
                        "area":  area,
                        "format": "netcdf",
                    },
                    outfile,
                )
                print(f"  ✓ Saved → {outfile}")
            except Exception as e:
                print(f"  ⚠ Failed for {year}-{month_str}: {e}")


# ── Synthetic / demo data generator ──────────────────────────────
def generate_synthetic_era5(cfg):
    """Create small synthetic NetCDF files (for pipeline testing)."""
    import xarray as xr

    region = cfg["region"]
    res = cfg["grid"]["resolution"]
    lats = np.arange(region["lat_min"], region["lat_max"] + res, res)
    lons = np.arange(region["lon_min"], region["lon_max"] + res, res)

    out_dir = os.path.join(ROOT, cfg["paths"]["raw_data"], "era5")
    os.makedirs(out_dir, exist_ok=True)

    start_year = int(cfg["time"]["start"][:4])
    end_year   = int(cfg["time"]["end"][:4])

    np.random.seed(42)
    for year in range(start_year, end_year + 1):
        times = np.arange(
            np.datetime64(f"{year}-01-01"),
            np.datetime64(f"{year+1}-01-01"),
            np.timedelta64(1, "h"),
        )
        # Keep only every 6th hour to reduce file size
        times = times[::6]

        shape = (len(times), len(lats), len(lons))

        ds = xr.Dataset(
            {
                "t2m":   (["time", "latitude", "longitude"],
                          273.15 + 15 + 20 * np.random.rand(*shape).astype(np.float32)),
                "d2m":   (["time", "latitude", "longitude"],
                          273.15 + 5 + 15 * np.random.rand(*shape).astype(np.float32)),
                "u10":   (["time", "latitude", "longitude"],
                          5 * np.random.rand(*shape).astype(np.float32)),
                "v10":   (["time", "latitude", "longitude"],
                          5 * np.random.rand(*shape).astype(np.float32)),
                "tp":    (["time", "latitude", "longitude"],
                          0.001 * np.random.rand(*shape).astype(np.float32)),
                "swvl1": (["time", "latitude", "longitude"],
                          0.1 + 0.3 * np.random.rand(*shape).astype(np.float32)),
            },
            coords={
                "time":      times,
                "latitude":  lats,
                "longitude": lons,
            },
        )

        outfile = os.path.join(out_dir, f"era5_{year}.nc")
        ds.to_netcdf(outfile)
        print(f"  ✓ Synthetic ERA5 → {outfile}  ({ds.dims})")


# ── CLI ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download ERA5 data")
    parser.add_argument("--synthetic", action="store_true",
                        help="Generate synthetic demo data instead of real download")
    args = parser.parse_args()

    cfg = load_config()
    print("=" * 60)
    print("ERA5 Data Download")
    print("=" * 60)

    if args.synthetic:
        print("[MODE] Generating synthetic ERA5 data …")
        generate_synthetic_era5(cfg)
    else:
        print("[MODE] Downloading real ERA5 data via CDS API …")
        download_era5(cfg)

    print("Done.\n")
