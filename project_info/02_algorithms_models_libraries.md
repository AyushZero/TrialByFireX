# Algorithms, Models & Libraries

## 🧠 Machine Learning Models

### 1. Physics-Guided Logistic Regression ⭐ (Primary Model)
- **Type**: Binary classifier (1 feature: R_phys → p_ign)
- **Parameters**: 2 (slope `a`, intercept `b`)
- **Objective**: Binary cross-entropy (log-loss)
- **Class handling**: Balanced class weights + SMOTE variant
- **Interpretability**: Fully interpretable — single physics index
- **Library**: `sklearn.linear_model.LogisticRegression`

### 2. Attribute-Only Logistic Regression (Baseline)
- **Type**: Binary classifier (9 features → p_ign)
- **Parameters**: 10 (9 coefficients + intercept)
- **Purpose**: Show that physics composition adds value over raw features

### 3. Random Forest (Baseline)
- **Type**: Ensemble of 200 decision trees, max depth 10
- **Parameters**: ~10,000+ (tree splits)
- **Purpose**: Strong ML baseline for discrimination comparison
- **Library**: `sklearn.ensemble.RandomForestClassifier`

### 4. XGBoost (Baseline)
- **Type**: Gradient boosted trees, 200 rounds, depth 6, lr=0.1
- **Parameters**: ~5,000+ (tree parameters)
- **Class handling**: `scale_pos_weight` for imbalance
- **Purpose**: State-of-the-art ML baseline
- **Library**: `xgboost.XGBClassifier`

### 5. Weather-Only Logistic Regression (Ablation Baseline)
- **Type**: Logistic regression on T, RH, wind only
- **Purpose**: Isolate weather contribution vs vegetation + fire history

### 6. Canadian FWI Logistic Regression (Domain Baseline)
- **Type**: Simplified FWI → logistic regression
- **Purpose**: Compare against the operational fire weather standard
- **Reference**: Van Wagner (1987), Forestry Technical Report 35

---

## 📊 Analysis Algorithms

### Ablation Study
- **Method**: Systematically replace each physics factor with a neutral value (1 or 0)
- **Variants**: Full, No-T, No-F_avail, No-F_dry, No-G_spread, No-H_history
- **Metric**: ΔAUC-ROC from full R_phys

### SMOTE (Class Imbalance)
- **Algorithm**: Synthetic Minority Over-sampling Technique
- **Combined with**: Tomek link removal + random undersampling
- **Target ratio**: 10% minority class (from ~0.25%)
- **Library**: `imblearn.combine.SMOTETomek`, `imblearn.over_sampling.SMOTE`

### SHAP (Feature Importance)
- **Algorithm**: SHapley Additive exPlanations
- **Explainer**: `TreeExplainer` for RF and XGBoost
- **Output**: Per-feature importance + direction of contribution
- **Library**: `shap`

### Spatial Cross-Validation
- **Method**: 5-fold block-based CV (2° × 2° spatial blocks)
- **Purpose**: Test geographic generalisation
- **Implementation**: Custom block assignment + fold rotation

### Canadian FWI
- **Components**: FFMC (fine fuel moisture) → ISI (initial spread index)
- **Simplification**: Single-day approximation (full FWI needs multi-day DMC/DC)
- **Implementation**: `src/fwi.py`

---

## 📐 Physics-Guided Feature Engineering

| Factor | Variables | Physical Basis |
|--------|-----------|----------------|
| **F_avail** | NDVI | Live fuel mass via vegetation greenness (Pyne 1996) |
| **F_dry** | NDWI, RH, SM | Live Fuel Moisture Content proxy – canopy water, atmo dryness, soil deficit |
| **G_spread** | Wind, Slope | Rothermel (1972) fire spread model – wind and slope amplify rate of spread |
| **H_history** | FRP, hotspot count | Persistence – fires cluster spatially and temporally (Coen et al. 2018) |
| **R_phys** | All above + T | Multiplicative structure mirrors Canadian FWI logic |

---

## 📚 Libraries Used

### Core Data Processing
| Library | Version | Purpose |
|---------|---------|---------|
| `numpy` | ≥1.24 | Array operations, numerical computation |
| `pandas` | ≥2.0 | DataFrames, temporal operations |
| `xarray` | ≥2023.1 | NetCDF handling, gridded data |
| `netCDF4` | ≥1.6 | NetCDF I/O backend |
| `scipy` | ≥1.11 | Spatial operations, interpolation |

### Machine Learning
| Library | Version | Purpose |
|---------|---------|---------|
| `scikit-learn` | ≥1.3 | Logistic regression, Random Forest, metrics |
| `xgboost` | ≥2.0 | Gradient boosting baseline |
| `imbalanced-learn` | ≥0.11 | SMOTE, class imbalance handling |
| `shap` | ≥0.42 | Feature importance explanation |

### Geospatial
| Library | Version | Purpose |
|---------|---------|---------|
| `rasterio` | ≥1.3 | GeoTIFF I/O |
| `geopandas` | ≥0.14 | Spatial DataFrames |

### Visualization
| Library | Version | Purpose |
|---------|---------|---------|
| `matplotlib` | ≥3.7 | Static plots, publication figures |
| `seaborn` | ≥0.12 | Statistical visualization, heatmaps |
| `plotly` | ≥5.18 | Interactive charts in dashboard |

### Dashboard & API
| Library | Version | Purpose |
|---------|---------|---------|
| `streamlit` | ≥1.30 | Web dashboard framework |
| `folium` | ≥0.15 | Interactive maps |

### Data Access
| Library | Version | Purpose |
|---------|---------|---------|
| `cdsapi` | ≥0.6 | ERA5 data from Copernicus CDS |
| `requests` | ≥2.31 | NASA FIRMS REST API calls |
