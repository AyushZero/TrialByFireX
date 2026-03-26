# TrialsByFireX — Physics-Guided Ignition Probability Modelling

> Using ERA5, MODIS, Sentinel, SRTM, and FIRMS to produce calibrated daily wildfire ignition probabilities.

## Quick Start

```bash
# 1. Create environment
pip install -r requirements.txt

# 2. Configure CDS API (ERA5)
#    Place your .cdsapirc in your home directory (see scripts/download_era5.py)

# 3. Download data
python scripts/download_era5.py
python scripts/download_firms.py
python scripts/download_modis.py
python scripts/download_srtm.py

# 4. Preprocess & build features
python run_preprocess.py

# 5. Train & evaluate (see notebooks/ for interactive version)
python run_train.py

# 6. Inference for a specific day
python run_inference.py --date 2023-06-15
```

## Project Structure

| Directory | Purpose |
|-----------|---------|
| `scripts/` | Data download scripts (ERA5, FIRMS, MODIS, SRTM) |
| `src/` | Core library (grid, preprocess, features, models, evaluate) |
| `notebooks/` | Jupyter notebooks for exploration and modelling |
| `data/` | Raw, processed, and feature data |
| `models/` | Saved model artifacts |
| `outputs/` | Probability maps and visualizations |

## Study Region & Period

- **Region**: California (32°N–42°N, 124°W–114°W)
- **Period**: 2021–2023
- **Grid**: 0.25° (~28 km)

## Documentation

See `Documentation.txt` for the full system architecture, data flow, math specification, and engineering plan.
