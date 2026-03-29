#!/usr/bin/env python3
"""
Create two full-grid demo daily datasets (2021-2023 window) for website testing.

Outputs:
  data/demo/daily_grid_2022-07-15.csv
  data/demo/daily_grid_2023-07-15.csv
"""

import os
import sys

import numpy as np
import pandas as pd
import xarray as xr
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def export_one_date(ds, date_str, out_path):
    t_idx = int(np.argmin(np.abs(ds.time.values - np.datetime64(date_str))))
    day = ds.isel(time=t_idx)

    lats = day.latitude.values
    lons = day.longitude.values
    lat_grid, lon_grid = np.meshgrid(lats, lons, indexing="ij")

    cols = [
        "t_max", "rh_min", "u10_max", "sm_top",
        "ndvi", "ndwi", "slope", "frp_hist", "count_hist", "ignition",
    ]

    frame = {
        "date": np.full(lat_grid.size, pd.to_datetime(day.time.values).strftime("%Y-%m-%d"), dtype=object),
        "latitude": lat_grid.reshape(-1),
        "longitude": lon_grid.reshape(-1),
    }

    for c in cols:
        frame[c] = day[c].values.reshape(-1)

    out = pd.DataFrame(frame)
    out.to_csv(out_path, index=False)
    return out


def main():
    with open(os.path.join(ROOT, "config.yaml"), "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    features_path = os.path.join(ROOT, cfg["paths"]["features"], "features.nc")
    ds = xr.open_dataset(features_path)

    out_dir = os.path.join(ROOT, "data", "demo")
    os.makedirs(out_dir, exist_ok=True)

    f1 = os.path.join(out_dir, "daily_grid_2022-07-15.csv")
    f2 = os.path.join(out_dir, "daily_grid_2023-07-15.csv")

    d1 = export_one_date(ds, "2022-07-15", f1)
    d2 = export_one_date(ds, "2023-07-15", f2)

    ds.close()

    readme_path = os.path.join(out_dir, "README_DAILY_GRID.md")
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(
            "# Daily Grid Demo Data\n\n"
            "These files contain full California grid data for one day each.\n"
            "Feature columns are normalized in [0,1] and include ignition truth.\n\n"
            f"- daily_grid_2022-07-15.csv (rows={len(d1)})\n"
            f"- daily_grid_2023-07-15.csv (rows={len(d2)})\n"
        )

    print("Created:")
    print(f"  {f1}")
    print(f"  {f2}")


if __name__ == "__main__":
    main()
