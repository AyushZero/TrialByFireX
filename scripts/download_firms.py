#!/usr/bin/env python3
"""
Download FIRMS (VIIRS) active-fire data for the study region.

Uses the NASA FIRMS REST API.
API key is read from config.yaml: firms -> map_key.

Usage:
    python scripts/download_firms.py                 # real download
    python scripts/download_firms.py --synthetic      # generate demo data
"""

import argparse, os, sys, yaml
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def load_config():
    with open(os.path.join(ROOT, "config.yaml")) as f:
        return yaml.safe_load(f)


# Real FIRMS download
def download_firms(cfg):
    """Download VIIRS fire data from NASA FIRMS REST API."""
    import requests

    region = cfg["region"]
    map_key = cfg["firms"]["map_key"]
    out_dir = os.path.join(ROOT, cfg["paths"]["raw_data"], "firms")
    os.makedirs(out_dir, exist_ok=True)

    start_year = int(cfg["time"]["start"][:4])
    end_year   = int(cfg["time"]["end"][:4])

    # FIRMS API: area query for VIIRS_SNPP_NRT or VIIRS_SNPP_SP
    # The API supports date ranges; we download year-by-year
    base_url = "https://firms.modaps.eosdis.nasa.gov/api/area/csv"
    source = "VIIRS_SNPP_SP"  # Standard Processing (archival)

    for year in range(start_year, end_year + 1):
        outfile = os.path.join(out_dir, f"firms_{year}.csv")
        if os.path.exists(outfile):
            print(f"  [skip] {outfile} already exists")
            continue

        # FIRMS API supports max 5-day ranges for archival area queries
        all_data = []
        from datetime import date, timedelta
        d = date(year, 1, 1)
        end_d = date(year, 12, 31)

        while d <= end_d:
            chunk_end = min(d + timedelta(days=4), end_d)
            n_days = (chunk_end - d).days + 1

            url = (
                f"{base_url}/{map_key}/{source}/"
                f"{region['lon_min']},{region['lat_min']},"
                f"{region['lon_max']},{region['lat_max']}/"
                f"{n_days}/{d.strftime('%Y-%m-%d')}"
            )

            print(f"    Fetching {d} to {chunk_end}...")
            resp = requests.get(url, timeout=120)
            if resp.status_code == 200 and len(resp.text.strip()) > 0:
                lines = resp.text.strip().split("\n")
                if len(lines) > 1:  # header + data
                    if not all_data:
                        all_data.append(lines[0])  # header
                    all_data.extend(lines[1:])
            else:
                print(f"    [WARN] Status {resp.status_code} for {d}")

            d = chunk_end + timedelta(days=1)

        if all_data:
            with open(outfile, "w") as f:
                f.write("\n".join(all_data))
            print(f"  [OK] Saved to {outfile} ({len(all_data)-1} records)")
        else:
            print(f"  [WARN] No data retrieved for {year}")


# Synthetic data
def generate_synthetic_firms(cfg):
    """Create synthetic FIRMS-like CSV files for pipeline testing."""
    region = cfg["region"]
    out_dir = os.path.join(ROOT, cfg["paths"]["raw_data"], "firms")
    os.makedirs(out_dir, exist_ok=True)

    start_year = int(cfg["time"]["start"][:4])
    end_year   = int(cfg["time"]["end"][:4])

    np.random.seed(123)
    for year in range(start_year, end_year + 1):
        n_fires = np.random.randint(800, 2500)

        dates = pd.date_range(f"{year}-05-01", f"{year}-11-30", freq="D")
        fire_dates = np.random.choice(dates, size=n_fires)

        df = pd.DataFrame({
            "latitude":   np.random.uniform(region["lat_min"], region["lat_max"], n_fires),
            "longitude":  np.random.uniform(region["lon_min"], region["lon_max"], n_fires),
            "acq_date":   [pd.Timestamp(d).strftime("%Y-%m-%d") for d in fire_dates],
            "acq_time":   np.random.randint(0, 2400, n_fires),
            "confidence": np.random.choice(["nominal", "high", "low"], n_fires,
                                           p=[0.5, 0.35, 0.15]),
            "frp":        np.random.exponential(15.0, n_fires).round(1),
            "type":       np.random.choice(["vegetation", "other"], n_fires,
                                           p=[0.85, 0.15]),
        })

        outfile = os.path.join(out_dir, f"firms_{year}.csv")
        df.to_csv(outfile, index=False)
        print(f"  [OK] Synthetic FIRMS -> {outfile} ({n_fires} fires)")


# Command-line interface
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download FIRMS data")
    parser.add_argument("--synthetic", action="store_true",
                        help="Generate synthetic demo data")
    args = parser.parse_args()

    cfg = load_config()
    print("=" * 60)
    print("FIRMS VIIRS Fire Data Download")
    print("=" * 60)

    if args.synthetic:
        print("[MODE] Generating synthetic FIRMS data...")
        generate_synthetic_firms(cfg)
    else:
        print("[MODE] Downloading real FIRMS data...")
        download_firms(cfg)

    print("Done.\n")
