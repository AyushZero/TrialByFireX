# Physics-Guided Ignition Probability Model Architecture

## 1. Overview
The core objective of the model is to estimate the daily probability $p_{\text{ign}}(t) \in [0,1]$ of a wildfire ignition in a given $0.25^\circ \times 0.25^\circ$ spatial grid cell.

Unlike black-box machine learning approaches (e.g., Random Forests, XGBoost), this model utilizes a **Physics-Guided Logistic Regression**. It synthesizes deeply understood fire science principles into a single interpretable "Risk Index" ($R_{phys}$), which is then rigorously calibrated to probabilities.

## 2. Input Data Sources (Features)
The model consumes 9 distinct raw data layers spanning four primary domains:

| Variable | Source | Spatial Res | Temporal Res | Physical Meaning |
| :--- | :--- | :--- | :--- | :--- |
| `t_max` | ERA5 Reanalysis | $0.25^\circ$ | Daily Max | Evaporative demand |
| `rh_min` | ERA5 Reanalysis | $0.25^\circ$ | Daily Min | Atmospheric dryness |
| `u10_max` | ERA5 Reanalysis | $0.25^\circ$ | Daily Max | Wind-driven spread |
| `sm_top` | ERA5 Reanalysis | $0.25^\circ$ | Daily Mean| Soil moisture deficit |
| `ndvi` | MODIS/Sentinel | 250m $\to 0.25^\circ$ | 16-day | Live fuel abundance |
| `ndwi` | MODIS/Sentinel | 250m $\to 0.25^\circ$ | 16-day | Canopy water deficit |
| `slope` | SRTM DEM | 30m $\to 0.25^\circ$ | Static | Terrain-driven spread |
| `frp_hist` | FIRMS VIIRS | 375m $\to 0.25^\circ$ | 7-day trailing | Fire intensity history |
| `count_hist`| FIRMS VIIRS | 375m $\to 0.25^\circ$ | 7-day trailing | Ignition density history |

*(NB: All features are $Z$-score normalized before being passed into the physics formulations below, yielding standardized variables $\tilde{X}$.)*

## 3. The Physics Index ($R_{phys}$)

The model architecture defines fire risk as a multiplicative synthesis of four necessary conditions: **Fuel Availability, Fuel Dryness, Spread Potential, and Historical Activity**.

### A. Fuel Availability ($F_{avail}$)
Fire requires combustible material. We use the Normalized Difference Vegetation Index (NDVI) as a proxy for above-ground biomass.
$$F_{avail} = \max(0, \tilde{\text{NDVI}})$$

### B. Fuel Dryness ($F_{dry}$)
Dryness is a multi-dimensional state encompassing atmospheric ($\text{RH}$), canopy ($\text{NDWI}$), and subterranean ($\text{SM}$) conditions.
$$F_{dry} = \alpha_1(1 - \tilde{\text{NDWI}}) + \alpha_2(1 - \tilde{\text{RH}}_{min}) + \alpha_3(1 - \tilde{\text{SM}}_{top})$$
*(Weights $\alpha_i$ sum to 1. Default: `[0.4, 0.3, 0.3]`)*

### C. Spread Potential ($G_{spread}$)
Fires naturally accelerate uphill and with high winds. We model this as an amplification factor (base 1.0).
$$G_{spread} = 1 + \beta_1 \cdot \tilde{U}_{10} + \beta_2 \cdot \tilde{\theta}_{slope}$$
*(Weights $\beta_i$ sum to 1. Default: `[0.6, 0.4]`)*

### D. Historical Patterning ($H_{history}$)
Recent nearby ignitions vastly increase the prior probability of secondary ignitions or smoldering holdovers.
$$H_{history} = \gamma_1 \cdot \tilde{\text{FRP}}_{hist} + \gamma_2 \cdot \tilde{\text{Count}}_{hist}$$
*(Weights $\gamma_i$ sum to 1. Default: `[0.6, 0.4]`)*

### Composite Index
The final index synthesizes these components. Assuming a fire requires fuel, dryness, and spread potential to ignite, plus an additive bias from local history:
$$R_{phys} = \left( F_{avail} \cdot F_{dry} \cdot G_{spread} \right) + H_{history}$$

## 4. Probability Calibration (Logistic Regression)

The $R_{phys}$ score is uncalibrated. We train a Logistic Regression model to inherently handle class imbalance and map the physical score to a highly calibrated probability $p_{ign} \in [0, 1]$.

$$ p_{ign} = \frac{1}{1 + e^{-(w_0 + w_1 \cdot R_{phys})}} $$

Because the model only learns two parameters ($w_0$ intercept, $w_1$ coefficient), it is entirely immune to the severe overfitting seen in deep learning approaches on imbalanced spatial wildfire data, ensuring immense geographic extrapolation capability.

## 5. Model Evaluation Metrics

1. **AUC-ROC (Area Under Receiver Operating Characteristic)**: Measures the model's ability to discriminate between ignition and non-ignition days, regardless of class imbalance limit.
2. **AUC-PR (Area Under Precision-Recall Curve)**: Heavily penalizes false alarms (False Positives) on rare events, prioritizing precision.
3. **Brier Score**: Measures probability calibration. Important for operational confidence (e.g., if the model says 80%, it should happen 80% of the time).
4. **Log-Loss (Binary Cross-Entropy)**: Evaluates the severity of divergence between the predicted probability and the true binary label.
5. **Spatial Generalization (K-Means Spatial CV)**: Validated using geographically disparate 5-fold blocks rather than just temporally.
