# Using the Trained Model – User Input & Output

## 🖥️ Dashboard Usage (production_dashboard.py)

### How to Launch
```bash
cd c:\Minor\TrialsByFireX
streamlit run production_dashboard.py
```
Opens at: http://localhost:8501

---

## 📥 User Inputs (Dashboard)

The user provides current weather, vegetation, terrain, and fire history conditions via interactive sliders:

### Meteorological Conditions
| Input | Range | Units | Typical Value |
|-------|-------|-------|---------------|
| Temperature | 0–50 | °C | 35 (summer) |
| Relative Humidity | 0–100 | % | 20 (dry) |
| Wind Speed | 0–30 | m/s | 8 |
| Precipitation | 0–50 | mm | 0 |
| Soil Moisture | 0–0.5 | m³/m³ | 0.15 |

### Vegetation & Terrain
| Input | Range | Units | Typical Value |
|-------|-------|-------|---------------|
| NDVI | 0–1.0 | dimensionless | 0.45 |
| NDWI | 0–1.0 | dimensionless | 0.30 |
| Slope | 0–45 | degrees | 10 |

### Fire History (last 7 days)
| Input | Range | Units | Typical Value |
|-------|-------|-------|---------------|
| FRP History | 0–100 | MW | 0 (no recent fire) |
| Fire Count | 0–20 | count | 0 |

---

## 📤 User Outputs (Dashboard)

### 1. Risk Level
- **Visual**: Large colored box showing LOW (green) / MODERATE (orange) / HIGH (red)
- **Numeric**: Ignition probability as percentage (e.g., "47.3%")

### 2. Component Breakdown
| Component | Description |
|-----------|-------------|
| R_phys | Composite physics risk index |
| F_avail | Fuel availability score |
| F_dry | Fuel dryness score |
| G_spread | Wind-slope spread score |
| H_history | Fire history score |

### 3. Risk Gauge
- Plotly gauge dial showing probability 0–100%
- Color zones: green (0–30%), orange (30–60%), red (60–100%)

### 4. Risk Map (separate tab)
- Date-selectable heatmap over California grid
- Dark-themed Plotly heatmap with lat/lon
- Alert threshold slider
- Statistics: mean p_ign, max p_ign, cells above threshold

### 5. Analysis Results (separate tab)
- ROC/PR curves, confusion matrices, SHAP plots
- Ablation study results, threshold analysis
- Seasonal performance breakdown

---

## 🔧 Programmatic Usage (run_inference.py)

### Command Line
```bash
python run_inference.py --date 2023-07-15
```

### Inputs
| Parameter | Source |
|-----------|--------|
| Date | Command line argument (`--date YYYY-MM-DD`) |
| Weather data | Loaded from `data/raw/era5/` |
| Vegetation data | Loaded from `data/raw/modis/` |
| Terrain data | Loaded from `data/raw/srtm/` |
| Fire history | Loaded from `data/raw/firms/` |
| Model | Loaded from `models/physics_logistic.pkl` |
| Norm params | Loaded from `data/processed/norm_params.json` |

### Outputs
| File | Format | Description |
|------|--------|-------------|
| `outputs/p_ign_YYYY-MM-DD.nc` | NetCDF | 2D probability grid (lat × lon) |
| `outputs/map_YYYY-MM-DD.png` | PNG | Visualization of probability map |

---

## 📊 Pipeline Usage (run_train.py)

### Command
```bash
python run_preprocess.py [--synthetic]   # Step 1: data → features
python run_train.py                      # Step 2: features → models + evaluation
```

### Output Summary
- **7 trained models** saved as `.pkl` files in `models/`
- **25+ analysis plots** saved as `.png` in `outputs/`
- **Metric tables** saved as `.csv` in `outputs/`
- All plots + models are used by the dashboard automatically
