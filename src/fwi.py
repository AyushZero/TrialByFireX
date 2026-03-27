"""
Canadian Fire Weather Index (FWI) - simplified implementation.

Computes a weather-only fire danger index from ERA5 variables as a
baseline comparison for the physics-guided R_phys.

Reference: Van Wagner, C.E. (1987). "Development and Structure of the
Canadian Forest Fire Weather Index System." Forestry Technical Report 35.

Simplified for grid-cell daily data: uses T, RH, wind, precipitation
to compute a composite danger rating.
"""

import numpy as np


def compute_ffmc(temp_C, rh, wind_kmh, precip_mm, ffmc_prev=85.0):
    """
    Fine Fuel Moisture Code (simplified).
    Tracks moisture in fine surface fuels (1-2 cm litter).

    Parameters
    ----------
    temp_C : float/array - temperature ( degC)
    rh : float/array - relative humidity (%)
    wind_kmh : float/array - wind speed (km/h)
    precip_mm : float/array - 24h precipitation (mm)
    ffmc_prev : float - previous day FFMC (default 85 = standard start)
    """
    mo = 147.2 * (101.0 - ffmc_prev) / (59.5 + ffmc_prev)

    # Rain phase
    rf = np.where(precip_mm > 0.5, precip_mm - 0.5, 0.0)
    mo_r = np.where(
        rf > 0,
        np.where(mo <= 150.0,
                 mo + 42.5 * rf * np.exp(-100.0 / (251.0 - mo)) * (1.0 - np.exp(-6.93 / rf)),
                 mo + 42.5 * rf * np.exp(-100.0 / (251.0 - mo)) * (1.0 - np.exp(-6.93 / rf))
                 + 0.0015 * (mo - 150.0) ** 2 * rf ** 0.5),
        mo,
    )
    mo_r = np.minimum(mo_r, 250.0)

    # Drying/wetting phase
    Ed = 0.942 * rh ** 0.679 + 11.0 * np.exp((rh - 100.0) / 10.0) + 0.18 * (21.1 - temp_C) * (1.0 - np.exp(-0.115 * rh))
    Ew = 0.618 * rh ** 0.753 + 10.0 * np.exp((rh - 100.0) / 10.0) + 0.18 * (21.1 - temp_C) * (1.0 - np.exp(-0.115 * rh))

    k_d = 0.424 * (1.0 - (rh / 100.0) ** 1.7) + 0.0694 * wind_kmh ** 0.5 * (1.0 - (rh / 100.0) ** 8)
    k_w = 0.424 * (1.0 - ((100.0 - rh) / 100.0) ** 1.7) + 0.0694 * wind_kmh ** 0.5 * (1.0 - ((100.0 - rh) / 100.0) ** 8)

    k_d = k_d * 0.581 * np.exp(0.0365 * temp_C)
    k_w = k_w * 0.581 * np.exp(0.0365 * temp_C)

    m = np.where(mo_r > Ed, Ed + (mo_r - Ed) * 10.0 ** (-k_d),
         np.where(mo_r < Ew, Ew - (Ew - mo_r) * 10.0 ** (-k_w), mo_r))

    ffmc = 59.5 * (250.0 - m) / (147.2 + m)
    return np.clip(ffmc, 0.0, 101.0)


def compute_isi(ffmc, wind_kmh):
    """
    Initial Spread Index - combines FFMC with wind.
    """
    fm = 147.2 * (101.0 - ffmc) / (59.5 + ffmc)
    sf = 19.115 * np.exp(-0.1386 * fm) * (1.0 + fm ** 5.31 / (4.93e7))
    si = 0.208 * sf * np.exp(0.05039 * wind_kmh)
    return si


def compute_fwi_simple(temp_C, rh, wind_ms, precip_mm):
    """
    Simplified composite FWI score for daily grid data.

    This is a practical approximation — full FWI requires sequential
    daily computation. For a grid-based comparison baseline, we compute
    a single-day proxy.

    Parameters
    ----------
    temp_C : array - daily max temperature ( degC)
    rh : array - daily min relative humidity (%)
    wind_ms : array - daily max wind speed (m/s)
    precip_mm : array - daily total precipitation (mm)

    Returns
    -------
    fwi : array - fire weather index (higher = more dangerous)
    """
    wind_kmh = wind_ms * 3.6  # m/s -> km/h

    # Compute FFMC (single-day approximation starting from standard 85)
    ffmc = compute_ffmc(temp_C, rh, wind_kmh, precip_mm, ffmc_prev=85.0)

    # Compute ISI
    isi = compute_isi(ffmc, wind_kmh)

    # Simplified FWI ~ ISI * dryness factor
    # (full FWI uses DMC and DC which need multi-day accumulation)
    dryness = np.clip((100.0 - rh) / 100.0, 0, 1)
    temp_factor = np.clip(temp_C / 40.0, 0, 1)

    fwi = isi * (0.5 + 0.5 * dryness) * (0.5 + 0.5 * temp_factor)
    return np.clip(fwi, 0, None)


def normalize_fwi(fwi):
    """Normalise FWI to [0, 1] for use as a model feature."""
    fmin = np.nanmin(fwi)
    fmax = np.nanmax(fwi)
    if fmax == fmin:
        return np.zeros_like(fwi)
    return (fwi - fmin) / (fmax - fmin)


# -- Self-test -----------------------------------------------------
if __name__ == "__main__":
    # Test with typical summer California conditions
    t = np.array([35.0, 20.0, 40.0, 15.0])   #  degC
    rh = np.array([15.0, 60.0, 10.0, 80.0])   # %
    w = np.array([8.0, 2.0, 12.0, 1.0])       # m/s
    p = np.array([0.0, 5.0, 0.0, 10.0])       # mm

    fwi = compute_fwi_simple(t, rh, w, p)
    print(f"FWI values: {fwi.round(2)}")
    print(f"  Hot+dry+windy -> FWI={fwi[2]:.1f} (should be highest)")
    print(f"  Cool+wet+calm -> FWI={fwi[3]:.1f} (should be lowest)")
    assert fwi[2] > fwi[3], "FWI should be higher for hot/dry conditions"
    print("FWI self-test passed [OK]")
