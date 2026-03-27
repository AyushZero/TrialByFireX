# TrialsByFireX Usage Guide

A physics-guided wildfire ignition probability model for California.

## Table of Contents

1. [Environment Setup](#1-environment-setup)
2. [Data Download](#2-data-download)
3. [Preprocessing Pipeline](#3-preprocessing-pipeline)
4. [Model Training](#4-model-training)
5. [Evaluation and Validation](#5-evaluation-and-validation)
6. [Inference](#6-inference)
7. [Configuration Reference](#7-configuration-reference)
8. [Troubleshooting](#8-troubleshooting)

---

## 1. Environment Setup

### Prerequisites

- Python 3.8+
- Git
- ~10GB disk space for data

### Installation

```bash
# Clone repository
git clone <your-repo-url>
cd TrialsByFireX

# Create virtual environment
python -m venv venv

# Activate (Windows)
.\venv\Scripts\activate

# Activate (Linux/Mac)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Required Packages

Core dependencies (in `requirements.txt`):
```
numpy
pandas
xarray
netCDF4
scikit-learn
xgboost
lightgbm
imbalanced-learn
shap
matplotlib
seaborn
scipy
pyyaml
```

### API Keys Required

| Data Source | Registration URL | Config Location |
|-------------|------------------|-----------------|
| ERA5 (CDS) | https://cds.climate.copernicus.eu/ | `~/.cdsapirc` |
| FIRMS (NASA) | https://firms.modaps.eosdis.nasa.gov/ | `config.yaml → firms.map_key` |

#### Setting up CDS API (for ERA5)

1. Register at https://cds.climate.copernicus.eu/
2. Get your API key from your profile page
3. Create `~/.cdsapirc`:
```
url: https://cds.climate.copernicus.eu/api/v2
key: <your-uid>:<your-api-key>
```

---

## 2. Data Download

The project uses 4 data sources:

| Source | Variables | Resolution | Download Script |
|--------|-----------|------------|-----------------|
| ERA5 | Temperature, humidity, wind, precipitation, soil moisture | 0.25° hourly | `scripts/download_era5.py` |
| MODIS | NDVI, NDWI | 1km 16-day | `scripts/download_modis.py` |
| SRTM | Elevation, slope | 30m | `scripts/download_srtm.py` |
| FIRMS | Fire hotspots, FRP | Point data | `scripts/download_firms.py` |

### Download Commands

```bash
# Download ERA5 (requires CDS API key)
python scripts/download_era5.py

# Download FIRMS fire data (requires MAP key in config.yaml)
python scripts/download_firms.py

# MODIS - follow instructions printed by script
python scripts/download_modis.py

# SRTM - follow instructions printed by script
python scripts/download_srtm.py
```

### Data Directory Structure

After download, your data should look like:
```
data/
├── raw/
│   ├── era5/
│   │   ├── era5_2021_01.nc
│   │   ├── era5_2021_02.nc
│   │   └── ...
│   ├── firms/
│   │   ├── firms_2021.csv
│   │   └── ...
│   ├── modis/
│   │   ├── modis_2021.nc
│   │   └── ...
│   └── srtm/
│       └── srtm_slope.nc
├── processed/
└── features/
```

---

## 3. Preprocessing Pipeline

The preprocessing pipeline converts raw data to gridded daily features.

### Run Preprocessing

```bash
python run_preprocess.py
```

### What It Does

1. **ERA5 Processing**: Hourly → daily aggregates
   - `t_max`: Daily max temperature (°C)
   - `rh_min`: Daily min relative humidity (%)
   - `u10_max`: Daily max wind speed (m/s)
   - `p_tot`: Daily total precipitation (mm)
   - `sm_top`: Daily mean soil moisture (m³/m³)

2. **MODIS Processing**: 16-day composites → daily (forward-fill)
   - `ndvi`: Normalized Difference Vegetation Index
   - `ndwi`: Normalized Difference Water Index

3. **SRTM Processing**: Static terrain
   - `slope`: Terrain slope (degrees)

4. **FIRMS Processing**: Fire hotspots → grid labels + history
   - `ignition`: Binary fire occurrence (0/1)
   - `frp_hist`: Exponentially decayed fire history
   - `count_hist`: Fire count history

5. **Feature Engineering**: Physics-guided composites
   - `F_avail`: Fuel availability (from NDVI)
   - `F_dry`: Fuel dryness (from NDWI, RH, soil moisture)
   - `G_spread`: Spread potential (from wind, slope)
   - `H_history`: Fire history signal
   - `R_phys`: Composite ignition risk index

### Output

- `data/features/features.nc` - NetCDF with all features

---

## 4. Model Training

### Run Training

```bash
python run_train.py
```

### Models Trained (8 total)

| # | Model | Features | Purpose |
|---|-------|----------|---------|
| 1 | Physics Logistic | R_phys (1 feature) | Main physics-guided model |
| 2 | Physics + SMOTE | R_phys | With class balancing |
| 3 | Attribute Logistic | 9 raw features | Baseline without physics |
| 4 | Random Forest | 9 raw features | ML baseline |
| 5 | XGBoost | 9 raw features | Gradient boosting baseline |
| 5b | LightGBM | 9 raw features | Alternative boosting |
| 6 | Weather Only | 3 features | Weather-only baseline |
| 7 | FWI Baseline | FWI index | Canadian Fire Weather Index |

### Training Outputs

```
models/
├── physics_logistic.joblib
├── physics_smote.joblib
├── attribute_logistic.joblib
├── random_forest.joblib
├── xgboost.joblib
├── lightgbm.joblib
├── weather_logistic.joblib
└── fwi_logistic.joblib
```

---

## 5. Evaluation and Validation

### Metrics Generated

| Metric | Description |
|--------|-------------|
| AUC-ROC | Area under ROC curve |
| AUC-PR | Area under Precision-Recall curve |
| Brier Score | Probabilistic calibration |
| CSI | Critical Success Index (Threat Score) |
| GSS | Gilbert Skill Score (Equitable Threat Score) |
| Log-Loss | Binary cross-entropy |

### Evaluation Outputs

```
outputs/
├── model_comparison.csv        # All metrics for all models
├── roc_curves.png              # ROC curves comparison
├── pr_curves.png               # Precision-Recall curves
├── ablation_study.png          # Factor contribution analysis
├── ablation_results.csv        # Ablation numeric results
├── confusion_*.png             # Confusion matrices
├── shap_xgboost.png            # Feature importance
├── reliability_diagram.png     # Calibration plot
├── threshold_analysis.png      # F1 vs threshold
├── seasonal_analysis.png       # Performance by season
├── case_study_dixie_fire.png   # Dixie Fire probability map
└── ...
```

### Key Analyses

1. **Ablation Study**: Shows contribution of each factor (F_avail, F_dry, G_spread, H_history)
2. **Spatial Cross-Validation**: 5-fold geographic blocks to avoid spatial leakage
3. **SHAP Values**: Feature importance for tree-based models
4. **Stratified Performance**: Performance by slope/NDVI quartiles

---

## 6. Inference

### Generate Probability Maps

```bash
python run_inference.py --date 2023-06-15
```

### Output

- `outputs/probability_map_2023-06-15.png` - Spatial probability map

### Batch Inference

```python
from src.inference import run_inference
import yaml

with open("config.yaml") as f:
    cfg = yaml.safe_load(f)

# Generate maps for fire season
for date in ["2023-06-01", "2023-07-01", "2023-08-01"]:
    run_inference(cfg, date)
```

---

## 7. Configuration Reference

### config.yaml

```yaml
# Study Region
region:
  name: "California"
  lat_min: 32.0
  lat_max: 42.0
  lon_min: -124.0
  lon_max: -114.0

# Time Range
time:
  start: "2021-01-01"
  end: "2023-12-31"
  train_years: [2021]
  val_years: [2022]
  test_years: [2023]

# Grid Resolution
grid:
  resolution: 0.25  # degrees (~25km)

# Physics Weights (tunable)
alpha:           # F_dry weights
  alpha1: 0.4    # NDWI (canopy water)
  alpha2: 0.3    # RH (atmospheric)
  alpha3: 0.3    # SM (soil)

beta:            # G_spread weights
  beta1: 0.6     # Wind speed
  beta2: 0.4     # Slope

gamma:           # H_history weights
  gamma1: 0.6    # FRP history
  gamma2: 0.4    # Count history
```

### Physics Formula

```
R_phys = T̃ × F_avail × F_dry × G_spread + H_history

where:
  F_avail  = NDVI~
  F_dry    = α₁(1-NDWI~) + α₂(1-RH~) + α₃(1-SM~)
  G_spread = 1 + β₁U~ + β₂θ~
  H_history= γ₁FRP~_hist + γ₂Count~_hist
```

---

## 8. Troubleshooting

### Common Issues

#### ImportError: No module named 'xarray'
```bash
pip install xarray netCDF4
```

#### CDS API timeout
ERA5 downloads can take hours. The script automatically chunks by month and resumes.

#### Memory issues
Reduce the time range in config.yaml or process in smaller chunks.

#### FIRMS download fails
Check your MAP key in config.yaml. Register at https://firms.modaps.eosdis.nasa.gov/

#### SHAP analysis slow
For large datasets, SHAP samples are limited. You can reduce further by editing run_train.py.

### Getting Help

1. Check existing issues at the repository
2. Create a new issue with:
   - Error message
   - Python version
   - Operating system
   - Steps to reproduce

---

## Quick Start Summary

```bash
# 1. Setup
pip install -r requirements.txt

# 2. Configure API keys
# - Create ~/.cdsapirc for ERA5
# - Add FIRMS key to config.yaml

# 3. Download data
python scripts/download_era5.py
python scripts/download_firms.py
python scripts/download_modis.py
python scripts/download_srtm.py

# 4. Preprocess
python run_preprocess.py

# 5. Train & Evaluate
python run_train.py

# 6. Inference
python run_inference.py --date 2023-07-01

# 7. View results
ls outputs/
```
