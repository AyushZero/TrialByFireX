# Modelling Pipeline – Input, Output & Formulas

## 📥 Inputs to the Modelling Pipeline

### Raw Data Sources

| Data Source | Variables | Resolution | Format | Files |
|-------------|-----------|------------|--------|-------|
| **ERA5 Reanalysis** | 2m temperature, dewpoint, u10/v10 wind, total precip, soil moisture | 0.25° (lat/lon), hourly | NetCDF | `data/raw/era5/era5_{YEAR}.nc` |
| **MODIS** | NDVI, NDWI | 1km, 16-day composite | NetCDF | `data/raw/modis/modis_{YEAR}.nc` |
| **SRTM DEM** | Elevation → slope | 30m | NetCDF | `data/raw/srtm/srtm_slope.nc` |
| **FIRMS** | Lat, lon, date, FRP, confidence, type | 375m, daily | CSV | `data/raw/firms/firms_{YEAR}.csv` |

### Preprocessed Features (per grid cell per day)

| Feature | Derivation | Range |
|---------|-----------|-------|
| `t_max` | Daily max 2m temperature (°C), normalised | [0, 1] |
| `rh_min` | Daily min relative humidity (%, from Magnus formula), normalised | [0, 1] |
| `u10_max` | Daily max 10m wind speed (m/s), normalised | [0, 1] |
| `sm_top` | Daily mean top-layer soil moisture (m³/m³), normalised | [0, 1] |
| `ndvi` | NDVI (forward-filled from 16-day composite), normalised | [0, 1] |
| `ndwi` | NDWI/NDMI water index, normalised | [0, 1] |
| `slope` | Terrain slope (degrees), normalised | [0, 1] |
| `frp_hist` | Exponential-decayed FRP: Σ λ^k · FRP(t−k), normalised | [0, 1] |
| `count_hist` | Exponential-decayed hotspot count: Σ λ^k · y(t−k), normalised | [0, 1] |
| `ignition` | Binary label: 1 if fire in cell on day, else 0 | {0, 1} |

---

## 🧮 Formulas

### Normalisation (min-max)
```
X̃ = (X − X_min) / (X_max − X_min)    →    X̃ ∈ [0, 1]
```

### Relative Humidity (Magnus Formula)
```
γ(T, Td) = (17.625 × Td) / (243.04 + Td) − (17.625 × T) / (243.04 + T)
RH = 100 × exp(γ)
```

### Physics-Guided Composite Factors
```
F_avail(t)   = NDVĨ(t)                                           — fuel availability
F_dry(t)     = α₁(1 − NDWĨ) + α₂(1 − RH̃) + α₃(1 − SM̃)        — fuel dryness
G_spread(t)  = 1 + β₁·Ũ(t) + β₂·θ̃                              — wind-slope spread
H_history(t) = γ₁·FRP̃_hist + γ₂·Count̃_hist                      — fire history
```

### Composite Risk Index
```
R_phys(t) = T̃(t) · F_avail(t) · F_dry(t) · G_spread(t) + H_history(t)
```

### Logistic Calibration
```
p_ign(t) = σ(a · R_phys(t) + b) = 1 / (1 + exp(−(a · R_phys(t) + b)))
```

### Canadian FWI (simplified baseline)
```
FFMC → ISI → FWI = ISI × dryness_factor × temperature_factor
```

### SMOTE Resampling
```
Synthetic minority sample = x + rand(0,1) × (x_nn − x)
Target ratio: 10% fire class (was ~0.25%)
Combined with Tomek link removal and undersampling
```

---

## 📤 Outputs from Modelling

### Generated Files

| File | Description | Location |
|------|-------------|----------|
| `physics_logistic.pkl` | Trained physics-guided logistic model | `models/` |
| `physics_smote.pkl` | Physics model with SMOTE | `models/` |
| `attribute_logistic.pkl` | Attribute-only logistic model | `models/` |
| `random_forest.pkl` | Random Forest (200 trees) | `models/` |
| `xgboost.pkl` | XGBoost gradient boosting | `models/` |
| `weather_logistic.pkl` | Weather-only baseline | `models/` |
| `fwi_logistic.pkl` | Canadian FWI baseline | `models/` |
| `model_comparison.csv` | Full metrics table | `outputs/` |
| `roc_curves.png` | ROC curves – all models | `outputs/` |
| `pr_curves.png` | Precision-Recall curves | `outputs/` |
| `model_comparison_bar.png` | Bar chart comparison | `outputs/` |
| `confusion_*.png` | Confusion matrices (7 models) | `outputs/` |
| `shap_random_forest.png` | SHAP values – RF | `outputs/` |
| `shap_xgboost.png` | SHAP values – XGBoost | `outputs/` |
| `ablation_study.png` | Ablation study plot | `outputs/` |
| `ablation_results.csv` | Ablation metrics table | `outputs/` |
| `threshold_analysis.png` | F1/Precision/Recall vs threshold | `outputs/` |
| `reliability_diagram.png` | Calibration plot | `outputs/` |
| `seasonal_analysis.png` | Monthly/seasonal AUC | `outputs/` |
| `correlation_heatmap.png` | Feature correlation matrix | `outputs/` |
| `feature_distributions.png` | Histograms by class | `outputs/` |
| `R_phys_timeseries.png` | Temporal risk evolution | `outputs/` |
| `case_study_dixie_fire.png` | Case study probability map | `outputs/` |
| `performance_by_slope.csv` | Hit/FA rate by slope quartile | `outputs/` |
| `performance_by_ndvi.csv` | Hit/FA rate by NDVI quartile | `outputs/` |

### Evaluation Metrics

| Metric | Meaning | Direction |
|--------|---------|-----------|
| AUC-ROC | Discrimination – separating fire from no-fire | Higher ↑ |
| AUC-PR | Performance on rare events (fires) | Higher ↑ |
| Brier Score | Calibration quality of probabilities | Lower ↓ |
| Log-Loss | Cross-entropy loss | Lower ↓ |
| Spatial CV AUC | Generalisation across geographic blocks | Higher ↑ |
