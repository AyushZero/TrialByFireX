# Pipeline Outputs & Modelling Details

## 1. Algorithms & Libraries
**Core Language**: Python 3.11  
**Key Libraries**:
- *Data Engineering*: `xarray`, `pandas`, `numpy`, `rasterio`
- *Machine Learning*: `scikit-learn` (LogisticRegression, RandomForest), `xgboost` (XGBClassifier), `imbalanced-learn` (RandomUnderSampler)
- *Visualization*: `matplotlib`, `seaborn`, `shap`, `streamlit`, `folium`

**Models Trained (8 variations)**:
1. **Physics Logistic** (Final Model)
2. **Attribute Logistic** (Raw features, no physics)
3. **Random Forest** (Black-box ensemble)
4. **XGBoost** (Gradient boosting)
5. **FWI Proxy (Weather Only)** (Standard meteorological baseline)
6. *Ablation:* Physics (No Dryness)
7. *Ablation:* Physics (No Spread)
8. *Ablation:* Physics (No History)

## 2. Directory Structure & Outputs Received
Running `python run_train.py` systematically yields the following file outputs.

### A. Saved Models (`models/`)
Generated using `joblib` for seamless production deployment.
- `physics_logistic.pkl`: The primary production model.
- `attribute_logistic.pkl`, `random_forest.pkl`, `xgboost.pkl`, `fwi_baseline.pkl`: Baselines.

### B. Diagnostic Visualizations (`outputs/`)
| File | Description |
| :--- | :--- |
| **`feature_distributions.png`** | Kernel Density Estimation plots showing the divergence in feature distributions between non-fire days (blue) and fire days (red). |
| **`correlation_heatmap.png`** | Pearson correlation matrix identifying multicollinearity (e.g., negative correlation between Temp and RH). |
| **`model_comparison_bar.png`** | Bar chart comparing AUC-ROC, AUC-PR, Brier Score, and Log-Loss across all 8 models. |
| **`roc_curves.png`** & **`pr_curves.png`** | Trade-off curves highlighting Discrimination (ROC) and capability on severe class imbalances (PR). |
| **`confusion_*.png`** | 8 distinct confusion matrices (one per model). Shows True Positives vs False Positives at the default $0.5$ threshold. |
| **`threshold_analysis.png`** | Curves mapping Threshold vs F1-score to derive the mathematically optimal operational alert threshold. |
| **`reliability_diagram.png`** | Calibration curve proving that predicted physics probabilities map linearly to actual observed real-world frequencies. |
| **`shap_random_forest.png`** | SHAP beeswarm plot revealing the driving factors inside the black-box ensemble (usually highlighting NDVI and FRP). |
| **`seasonal_analysis.png`** | Timeseries breaking down physical risk distributions and model accuracy block-by-block through Spring, Summer, and Autumn. |
| **`R_phys_timeseries.png`** | Timeseries explicitly showing $R_{phys}$ escalating days or weeks prior to historical mass ignitions. |

### C. Tabular Reports (`outputs/`)
- **`model_comparison.csv`**: Raw metrics for all models (AUC, Brier, Log-Loss).
- **`performance_by_slope.csv`**: Stratified hit-rates proving the model works across flatland and steep terrain equally.
- **`performance_by_ndvi.csv`**: Stratified hit-rates proving performance across arid deserts and dense forests.
