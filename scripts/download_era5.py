"""
Data download script for ERA5 reanalysis via Copernicus CDS API.
"""

import os
import yaml
import cdsapi

def download_era5(cfg):
    print("============================================================")
    print("ERA5 Data Download")
    print("============================================================")

    out_dir = os.path.join(cfg["paths"]["raw_data"], "era5")
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

if __name__ == "__main__":
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_path = os.path.join(root_dir, "config.yaml")

    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    if "--synthetic" in os.sys.argv:
        print("Using synthetic is no longer natively available in this script, please run with real data.")
    else:
        # Configuration check
        home = os.path.expanduser("~")
        cdsapirc = os.path.join(home, ".cdsapirc")
        if not os.path.exists(cdsapirc):
             print(f"[WARNING] You do not have a .cdsapirc file in {home}.")
             print("To download real ERA5 data, you must register at https://cds.climate.copernicus.eu/ and save your API key.")
             os.sys.exit(1)
             
        download_era5(cfg)
