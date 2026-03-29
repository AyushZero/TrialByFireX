"""
Utilities for configurable physics formula profiles.

A profile can override alpha/beta/gamma weights and use a history-mixing
parameter to interpolate between additive and multiplicative history effects.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from src.features import (
    compute_F_avail,
    compute_F_dry,
    compute_G_spread,
    compute_H_history,
)


def _normalize_triplet(weights: Dict[str, float], keys):
    values = np.array([float(weights[k]) for k in keys], dtype=float)
    total = values.sum()
    if total <= 0:
        return {k: 1.0 / len(keys) for k in keys}
    values = values / total
    return {k: float(v) for k, v in zip(keys, values)}


def normalize_profile(profile: Dict) -> Dict:
    """Return a safe, normalized profile dict."""
    alpha = _normalize_triplet(profile["alpha"], ["alpha1", "alpha2", "alpha3"])
    beta = _normalize_triplet(profile["beta"], ["beta1", "beta2"])
    gamma = _normalize_triplet(profile["gamma"], ["gamma1", "gamma2"])
    mix_history = float(np.clip(profile.get("mix_history", 0.0), 0.0, 1.0))

    return {
        "name": profile.get("name", "profile"),
        "alpha": alpha,
        "beta": beta,
        "gamma": gamma,
        "mix_history": mix_history,
    }


def baseline_profile_from_cfg(cfg: Dict) -> Dict:
    """Build baseline profile from config weights."""
    return normalize_profile(
        {
            "name": "baseline",
            "alpha": cfg["alpha"],
            "beta": cfg["beta"],
            "gamma": cfg["gamma"],
            "mix_history": 0.0,
        }
    )


def merge_profile(cfg: Dict, profile: Optional[Dict]) -> Dict:
    """Merge optional profile over baseline config profile."""
    base = baseline_profile_from_cfg(cfg)
    if not profile:
        return base

    merged = {
        "name": profile.get("name", base["name"]),
        "alpha": dict(base["alpha"]),
        "beta": dict(base["beta"]),
        "gamma": dict(base["gamma"]),
        "mix_history": float(profile.get("mix_history", base["mix_history"])),
    }

    for group in ["alpha", "beta", "gamma"]:
        if group in profile and isinstance(profile[group], dict):
            merged[group].update(profile[group])

    return normalize_profile(merged)


def compute_r_phys_from_normed(
    normed_inputs: Dict[str, float],
    cfg: Dict,
    profile: Optional[Dict] = None,
) -> Dict[str, float]:
    """Compute components and R_phys using a merged formula profile."""
    p = merge_profile(cfg, profile)

    f_avail = compute_F_avail(normed_inputs["ndvi"])
    f_dry = compute_F_dry(
        normed_inputs["ndwi"],
        normed_inputs["rh_min"],
        normed_inputs["sm_top"],
        p["alpha"],
    )
    g_spread = compute_G_spread(normed_inputs["u10_max"], normed_inputs["slope"], p["beta"])
    h_history = compute_H_history(normed_inputs["frp_hist"], normed_inputs["count_hist"], p["gamma"])

    core = normed_inputs["t_max"] * f_avail * f_dry * g_spread
    mix = p["mix_history"]
    r_phys = (1.0 - mix) * (core + h_history) + mix * (core * (1.0 + h_history))

    return {
        "F_avail": f_avail,
        "F_dry": f_dry,
        "G_spread": g_spread,
        "H_history": h_history,
        "R_phys": r_phys,
        "mix_history": mix,
        "profile_name": p["name"],
        "alpha": p["alpha"],
        "beta": p["beta"],
        "gamma": p["gamma"],
    }
