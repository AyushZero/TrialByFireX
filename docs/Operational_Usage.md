# Operational Usage & Deployment Guide

## 1. Overview
This model translates multi-domain earth observation data (ERA5 atmosphere, FIRMS hot-spots, SRTM terrain, MODIS/Sentinel vegetation) into a unified, actionable probability $p_{ign} \in [0, 1]$ of daily fire ignition for a specific $0.25^\circ$ geographical cell.

It is designed to be fully operationalized via the `Production Dashboard`. 

## 2. Using the Final Model
### A. The Trained Artifact (`physics_logistic.pkl`)
The final output of the training pipeline is a calibrated `LogisticRegression` model, serialized to `models/physics_logistic.pkl`. It expects **exactly one specialized input**: the composite $R_{phys}$ score for a grid-cell-day.

### B. User Inputs (Operational Mode)
When a forecaster or first-responder interacts with the deployed model (via API or Dashboard), they **do not** interact with the Logistic Regression directly. They interact with the underlying physical parameters. 

**Required Inputs:**
- **Atmospheric**: Temperature (°C), Min Relative Humidity (%), Max Wind Speed (m/s), Average Volumetric Soil Moisture (fraction).
- **Vegetative**: Current NDVI (0-1), NDWI (0-1).
- **Geographic**: Cell Slope (degrees).
- **Historical**: Trailing 7-day Fire Radiative Power (MW), Ignition Count (N).

### C. Normalization & Parameterization
Because the model was trained on Z-score normalized data, real-time inputs are dynamically normalized using the historical statistical parameters saved in `data/processed/norm_params.json` during the data ingestion pipeline.

The Normalized values ($\tilde{X}$) are plugged through the tunable physics equations to generate $R_{phys}$:
$$ R_{phys} = \left(\max(0, \tilde{\text{NDVI}}) \cdot (\alpha_{dryness}) \cdot (\beta_{spread}) \right) + \gamma_{history} $$

### D. Output & Decision Thresholds
The output is the scalar $p_{ign}$.

**Interpreting $p_{ign}$**:
- Because ignitions are extremely rare (class imbalance), an "average" day might hold a baseline modeled probability of $<0.01\text{ (1%)}$.
- When conditions match historical fire-conducive extremes (e.g., Dixie Fire conditions), probabilities will exponentially elevate to $0.40 \text{ (40%)} \to 0.85 \text{ (85%)}$.
- **Optimal Alert Threshold**: The training pipeline specifically outputs a `threshold_analysis.png`, defining the optimal threshold maximizing the F1-Score. Operationally, any $p_{ign} > T_{optimal}$ should trigger an automated high-risk alert.

## 3. Libraries & Dependencies
The operational deployment context utilizes:
- **`scikit-learn`** (Model Inference)
- **`streamlit`** (Interactive Dashboard UI)
- **`folium` & `streamlit_folium`** (Map Visualization)
- **`pandas` & `numpy`** (On-the-fly math processing)

*Please refer to `Model_Architecture.md` for specific formulaic breakdowns.*
