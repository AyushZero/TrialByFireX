"""
Normalization - min-max normalization of continuous drivers to [0, 1].

Computes climatological min/max from multi-year data and persists
parameters for reproducible inference.
"""

import os, json
import numpy as np
import xarray as xr


def compute_norm_params(datasets, variables):
    """
    Compute global min and max for each variable across a dict of
    xr.Dataset / xr.DataArray objects.

    Parameters
    ----------
    datasets : dict[str, xr.DataArray]
        Variable name -> DataArray (all years concatenated).
    variables : list[str]
        Which keys in `datasets` to normalise.

    Returns
    -------
    params : dict  -  { var_name: {"min": float, "max": float} }
    """
    params = {}
    for var in variables:
        arr = datasets[var]
        vmin = float(np.nanmin(arr.values))
        vmax = float(np.nanmax(arr.values))
        if vmax == vmin:
            vmax = vmin + 1e-6  # avoid division by zero
        params[var] = {"min": vmin, "max": vmax}
        print(f"  {var:>12s}  min={vmin:+10.4f}  max={vmax:+10.4f}")
    return params


def normalize(data, params):
    """
    Apply min-max normalisation:  X = (X - Xmin) / (Xmax - Xmin)

    Parameters
    ----------
    data : xr.DataArray or ndarray
    params : dict  -  {"min": float, "max": float}

    Returns
    -------
    Normalised data (same type), clipped to [0, 1].
    """
    vmin = params["min"]
    vmax = params["max"]
    normed = (data - vmin) / (vmax - vmin)
    return np.clip(normed, 0.0, 1.0)


def save_params(params, path):
    """Save normalisation parameters to JSON."""
    with open(path, "w") as f:
        json.dump(params, f, indent=2)
    print(f"  [OK] Saved norm params -> {path}")


def load_params(path):
    """Load normalisation parameters from JSON."""
    with open(path) as f:
        return json.load(f)


# Self-test
if __name__ == "__main__":
    # Quick sanity check
    x = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
    p = {"min": 10.0, "max": 50.0}
    xn = normalize(x, p)
    assert np.allclose(xn, [0.0, 0.25, 0.5, 0.75, 1.0])
    print("Normalize self-test passed [OK]")
