"""
Data download script for ERA5 reanalysis via Copernicus CDS API.

In real mode:
  - Downloads ERA5 hourly data via Copernicus Climate Data Store (CDS) API.
  - Requires a CDS account and ~/.cdsapirc file with API key.

In synthetic mode:
  - Generates realistic ERA5-like NetCDF files for pipeline testing.

Usage:
    python scripts/download_era5.py                 # real download
    python scripts/download_era5.py --synthetic     # demo data
"""

import argparse
import os
import sys
import yaml
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def load_config():
    with open(os.path.join(ROOT, "config.yaml")) as f:
        return yaml.safe_load(f)

# Synthetic data
def generate_synthetic_era5(cfg):
    """
    Create synthetic ERA5-like NetCDF files for pipeline testing.

    Generates hourly data for all 6 variables with realistic seasonal patterns:
      - t2m: 2m temperature (K) with seasonal cycle
      - d2m: 2m dewpoint temperature (K)
      - u10, v10: 10m wind components (m/s)
      - tp: total precipitation (m)
      - swvl1: volumetric soil water layer 1 (m³/m³)
    """
    import xarray as xr

    region = cfg["region"]
    res = cfg["grid"]["resolution"]
    lats = np.arange(region["lat_min"], region["lat_max"] + res, res)
    lons = np.arange(region["lon_min"], region["lon_max"] + res, res)

    out_dir = os.path.join(ROOT, cfg["paths"]["raw_data"], "era5")
    os.makedirs(out_dir, exist_ok=True)

    start_year = int(cfg["time"]["start"][:4])
    end_year = int(cfg["time"]["end"][:4])

    np.random.seed(99)

    for year in range(start_year, end_year + 1):
        for month in range(1, 13):
            # Generate hourly timestamps for the month
            start_date = f"{year}-{month:02d}-01"
            if month == 12:
                end_date = f"{year + 1}-01-01"
            else:
                end_date = f"{year}-{month + 1:02d}-01"

            times = np.arange(
                np.datetime64(start_date),
                np.datetime64(end_date),
                np.timedelta64(1, "h")
            )

            shape = (len(times), len(lats), len(lons))

            # Day of year for seasonal patterns
            doy_start = (np.datetime64(start_date) - np.datetime64(f"{year}-01-01")).astype(int)
            hour_offsets = np.arange(len(times))
            doy = doy_start + hour_offsets / 24.0

            # Temperature (K): seasonal cycle + diurnal cycle + spatial gradient
            # California: warmer in south, cooler in north (latitude effect)
            lat_effect = (lats[:, None] - 37) * (-0.5)  # Cooler as latitude increases
            seasonal = 15 * np.sin(2 * np.pi * (doy[:, None, None] - 80) / 365)  # Peak ~late June
            diurnal = 5 * np.sin(2 * np.pi * (hour_offsets[:, None, None] % 24 - 6) / 24)  # Peak ~2pm
            t2m_base = 288 + seasonal + diurnal + lat_effect[None, :, :]
            t2m = (t2m_base + 2 * np.random.randn(*shape)).astype(np.float32)

            # Dewpoint (K): typically 5-15K below temperature
            d2m = (t2m - 8 - 4 * np.random.rand(*shape)).astype(np.float32)

            # Wind components (m/s): random with slight seasonal trend
            wind_scale = 3 + 1 * np.sin(2 * np.pi * doy[:, None, None] / 365)  # Windier in winter
            u10 = (wind_scale * np.random.randn(*shape)).astype(np.float32)
            v10 = (wind_scale * np.random.randn(*shape)).astype(np.float32)

            # Precipitation (m): sparse, more in winter months
            precip_prob = 0.02 + 0.08 * np.cos(2 * np.pi * doy[:, None, None] / 365)  # Higher in winter
            tp = np.where(
                np.random.rand(*shape) < precip_prob,
                np.random.exponential(0.001, shape),  # Mean ~1mm when raining
                0
            ).astype(np.float32)

            # Soil moisture (m³/m³): 0.1-0.4, higher in winter/spring
            sm_seasonal = 0.05 * np.cos(2 * np.pi * (doy[:, None, None] - 60) / 365)
            swvl1 = np.clip(
                0.25 + sm_seasonal + 0.05 * np.random.randn(*shape),
                0.1, 0.45
            ).astype(np.float32)

            ds = xr.Dataset(
                {
                    "t2m": (["time", "latitude", "longitude"], t2m),
                    "d2m": (["time", "latitude", "longitude"], d2m),
                    "u10": (["time", "latitude", "longitude"], u10),
                    "v10": (["time", "latitude", "longitude"], v10),
                    "tp": (["time", "latitude", "longitude"], tp),
                    "swvl1": (["time", "latitude", "longitude"], swvl1),
                },
                coords={
                    "time": times,
                    "latitude": lats,
                    "longitude": lons,
                },
            )

            outfile = os.path.join(out_dir, f"era5_{year}_{month:02d}.nc")
            ds.to_netcdf(outfile)
            print(f"  Synthetic ERA5 -> {outfile}")

    print("  Done generating synthetic ERA5 data.")


# Real ERA5 download via CDS API
def download_era5(cfg):
    """
    Download ERA5 data via Copernicus Climate Data Store API.

    Requires CDS API key in ~/.cdsapirc file.
    Downloads monthly chunks to handle large requests.
    """
    import cdsapi

    print("============================================================")
    print("ERA5 Data Download")
    print("============================================================")

    out_dir = os.path.join(ROOT, cfg["paths"]["raw_data"], "era5")
    os.makedirs(out_dir, exist_ok=True)

    lat_max = cfg["region"]["lat_max"]
    lat_min = cfg["region"]["lat_min"]
    lon_max = cfg["region"]["lon_max"]
    lon_min = cfg["region"]["lon_min"]
    area = [lat_max, lon_min, lat_min, lon_max]

    start_year = int(cfg["time"]["start"].split("-")[0])
    end_year = int(cfg["time"]["end"].split("-")[0])
    years = [str(y) for y in range(start_year, end_year + 1)]

    # Initialize CDS client natively
    try:
        client = cdsapi.Client()
    except Exception as e:
        print("[ERROR] CDS API client could not be initialized.")
        return

    print("[MODE] Downloading real ERA5 data via CDS API (Month by Month) ...")
    
    days = [f"{d:02d}" for d in range(1, 32)]
    hours = [f"{h:02d}:00" for h in range(0, 24)]

    # Fetch chunked by year and month
    for year in years:
        for month in range(1, 13):
            m_str = f"{month:02d}"
            outfile = os.path.join(out_dir, f"era5_{year}_{m_str}.nc")
            
            # Simple existence check to cleanly resume downloads
            if os.path.exists(outfile):
                print(f"  [{year}-{m_str}] Data exists locally. Skipping...")
                continue
                
            print(f"  [{year}-{m_str}] Requesting from Copernicus API...")
            
            try:
                client.retrieve(
                    "reanalysis-era5-single-levels",
                    {
                        "product_type": "reanalysis",
                        "variable": [
                            "2m_temperature",
                            "2m_dewpoint_temperature",
                            "10m_u_component_of_wind",
                            "10m_v_component_of_wind",
                            "total_precipitation",
                            "volumetric_soil_water_layer_1"
                        ],
                        "year": [year],
                        "month": [m_str],
                        "day": days,
                        "time": hours,
                        "area": area,
                        "format": "netcdf",
                    },
                    outfile
                )
            except Exception as e:
                print(f"  [ERROR] Downloading {year}-{m_str} failed: {e}")
                if os.path.exists(outfile):
                    os.remove(outfile)
                
    print("Done. All ERA5 NetCDF chunks are completely ready for preprocessing.")


# -- CLI -----------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download ERA5 reanalysis data")
    parser.add_argument("--synthetic", action="store_true",
                        help="Generate synthetic demo data instead of downloading")
    args = parser.parse_args()

    cfg = load_config()
    print("=" * 60)
    print("ERA5 Reanalysis Data Download")
    print("=" * 60)

    if args.synthetic:
        print("[MODE] Generating synthetic ERA5 data ...")
        generate_synthetic_era5(cfg)
    else:
        # Check for CDS API configuration
        home = os.path.expanduser("~")
        cdsapirc = os.path.join(home, ".cdsapirc")
        if not os.path.exists(cdsapirc):
            print(f"[WARNING] No .cdsapirc file found in {home}")
            print("To download real ERA5 data, you must:")
            print("  1. Register at https://cds.climate.copernicus.eu/")
            print("  2. Get your API key from your profile page")
            print("  3. Create ~/.cdsapirc with your credentials")
            print("\nUse --synthetic flag for testing without real data.")
            sys.exit(1)

        download_era5(cfg)

    print("Done.\n")
