"""
Physics-guided feature engineering.

Implements the composite factors and ignition risk index R_phys
as described in the paper specification.

Equations
─────────
  F_avail(t)  = NDVI~(t)
  F_dry(t)    = α₁(1 − NDWI~) + α₂(1 − RH~) + α₃(1 − SM~)
  G_spread(t) = 1 + β₁·U~ + β₂·θ~
  H_history(t)= γ₁·FRP~_hist + γ₂·Count~_hist
  R_phys(t)   = T~(t) · F_avail(t) · F_dry(t) · G_spread(t) + H_history(t)
"""

import numpy as np
import xarray as xr


def compute_F_avail(ndvi_norm):
    """
    Fuel availability factor.
    F_avail = NDVI~  (normalised NDVI ∈ [0, 1])
    """
    return ndvi_norm


def compute_F_dry(ndwi_norm, rh_norm, sm_norm, alpha):
    """
    Fuel dryness factor (live-fuel moisture content proxy).

    F_dry = α₁(1 − NDWI~) + α₂(1 − RH~) + α₃(1 − SM~)

    Parameters
    ----------
    ndwi_norm, rh_norm, sm_norm : array-like  [0, 1]
    alpha : dict with keys alpha1, alpha2, alpha3
    """
    a1, a2, a3 = alpha["alpha1"], alpha["alpha2"], alpha["alpha3"]
    return a1 * (1 - ndwi_norm) + a2 * (1 - rh_norm) + a3 * (1 - sm_norm)


def compute_G_spread(u_norm, slope_norm, beta):
    """
    Wind–slope spreading potential.

    G_spread = 1 + β₁·U~ + β₂·θ~

    Parameters
    ----------
    u_norm : array-like   normalised wind speed [0, 1]
    slope_norm : array-like   normalised slope [0, 1]
    beta : dict with keys beta1, beta2
    """
    b1, b2 = beta["beta1"], beta["beta2"]
    return 1.0 + b1 * u_norm + b2 * slope_norm


def compute_H_history(frp_hist_norm, count_hist_norm, gamma):
    """
    Recent fire activity.

    H_history = γ₁·FRP~_hist + γ₂·Count~_hist

    Parameters
    ----------
    frp_hist_norm, count_hist_norm : array-like  [0, 1]
    gamma : dict with keys gamma1, gamma2
    """
    g1, g2 = gamma["gamma1"], gamma["gamma2"]
    return g1 * frp_hist_norm + g2 * count_hist_norm


def compute_R_phys(T_norm, F_avail, F_dry, G_spread, H_history):
    """
    Composite physics-guided ignition risk index.

    R_phys = T~ · F_avail · F_dry · G_spread  +  H_history

    Properties:
      • Dimensionless
      • Monotonic in all ignition-promoting factors
      • Multiplicative structure mirrors FWI logic
    """
    return T_norm * F_avail * F_dry * G_spread + H_history


def build_all_features(normed_data, cfg):
    """
    Compute all physics-guided features from normalised data.

    Parameters
    ----------
    normed_data : dict of str → xr.DataArray
        Expected keys: t_max, rh_min, u10_max, sm_top,
                        ndvi, ndwi, slope, frp_hist, count_hist
    cfg : dict – full config

    Returns
    -------
    features : dict of str → xr.DataArray
        Keys: F_avail, F_dry, G_spread, H_history, R_phys
    """
    alpha = cfg["alpha"]
    beta  = cfg["beta"]
    gamma = cfg["gamma"]

    F_avail   = compute_F_avail(normed_data["ndvi"])
    F_dry     = compute_F_dry(normed_data["ndwi"], normed_data["rh_min"],
                              normed_data["sm_top"], alpha)
    G_spread  = compute_G_spread(normed_data["u10_max"], normed_data["slope"], beta)
    H_history = compute_H_history(normed_data["frp_hist"], normed_data["count_hist"], gamma)
    R_phys    = compute_R_phys(normed_data["t_max"], F_avail, F_dry, G_spread, H_history)

    return {
        "F_avail":   F_avail,
        "F_dry":     F_dry,
        "G_spread":  G_spread,
        "H_history": H_history,
        "R_phys":    R_phys,
    }


# ── Self-test ─────────────────────────────────────────────────────
if __name__ == "__main__":
    # Quick sanity: when all normalized inputs are 0.5
    alpha = {"alpha1": 0.4, "alpha2": 0.3, "alpha3": 0.3}
    beta  = {"beta1": 0.6, "beta2": 0.4}
    gamma = {"gamma1": 0.6, "gamma2": 0.4}

    F_a = compute_F_avail(0.5)
    F_d = compute_F_dry(0.5, 0.5, 0.5, alpha)
    G_s = compute_G_spread(0.5, 0.5, beta)
    H_h = compute_H_history(0.5, 0.5, gamma)
    R   = compute_R_phys(0.5, F_a, F_d, G_s, H_h)

    print(f"F_avail  = {F_a:.4f}")
    print(f"F_dry    = {F_d:.4f}")
    print(f"G_spread = {G_s:.4f}")
    print(f"H_history= {H_h:.4f}")
    print(f"R_phys   = {R:.4f}")
    assert 0 < R < 5, "R_phys should be a small positive number"
    print("Features self-test passed ✓")
