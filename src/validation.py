"""
Validation helpers for comparing point forecasts against FIRMS observations.
"""

from __future__ import annotations

import os
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score

from src.evaluate import compute_csi, compute_gilbert_skill_score
from src.grid import assign_to_grid, grid_info


def _filter_firms(df: pd.DataFrame, cfg: Dict) -> pd.DataFrame:
    allowed_conf = set(cfg["firms"].get("confidence_filter", ["nominal", "high"]))
    allowed_type = cfg["firms"].get("type_filter", "vegetation")

    out = df.copy()
    if "confidence" in out.columns:
        out = out[out["confidence"].astype(str).str.lower().isin({c.lower() for c in allowed_conf})]
    if "type" in out.columns and allowed_type:
        out = out[out["type"].astype(str).str.lower() == str(allowed_type).lower()]
    return out


def _load_firms_year(year: int, firms_dir: str) -> pd.DataFrame:
    path = os.path.join(firms_dir, f"firms_{year}.csv")
    if not os.path.exists(path):
        return pd.DataFrame()
    return pd.read_csv(path)


def build_cell_truth_series(
    start_date: str,
    horizon_days: int,
    latitude: float,
    longitude: float,
    cfg: Dict,
    firms_dir: str,
) -> pd.DataFrame:
    """
    Build day-level ignition truth for one grid cell from FIRMS detections.

    Returns columns: date, observed_ignition, observed_count, observed_frp.
    """
    gi = grid_info(cfg)
    cell_lat_idx, cell_lon_idx = assign_to_grid(
        latitude,
        longitude,
        gi["lat_min"],
        gi["lon_min"],
        gi["resolution"],
    )

    dates = pd.date_range(start=start_date, periods=horizon_days, freq="D")
    years = sorted(set(dates.year.tolist()))
    by_year = {year: _filter_firms(_load_firms_year(year, firms_dir), cfg) for year in years}

    rows: List[Dict] = []
    for dt in dates:
        firms = by_year.get(dt.year, pd.DataFrame())
        if firms.empty:
            rows.append(
                {
                    "date": dt.strftime("%Y-%m-%d"),
                    "observed_ignition": 0,
                    "observed_count": 0,
                    "observed_frp": 0.0,
                }
            )
            continue

        day_df = firms[firms["acq_date"].astype(str) == dt.strftime("%Y-%m-%d")]
        if day_df.empty:
            rows.append(
                {
                    "date": dt.strftime("%Y-%m-%d"),
                    "observed_ignition": 0,
                    "observed_count": 0,
                    "observed_frp": 0.0,
                }
            )
            continue

        lat_idx, lon_idx = assign_to_grid(
            day_df["latitude"].to_numpy(),
            day_df["longitude"].to_numpy(),
            gi["lat_min"],
            gi["lon_min"],
            gi["resolution"],
        )
        in_cell = (lat_idx == cell_lat_idx) & (lon_idx == cell_lon_idx)
        cell_hits = day_df[in_cell]

        rows.append(
            {
                "date": dt.strftime("%Y-%m-%d"),
                "observed_ignition": int(len(cell_hits) > 0),
                "observed_count": int(len(cell_hits)),
                "observed_frp": float(cell_hits["frp"].sum()) if len(cell_hits) > 0 else 0.0,
            }
        )

    return pd.DataFrame(rows)


def evaluate_point_forecast(
    forecast_df: pd.DataFrame,
    truth_df: pd.DataFrame,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """Evaluate a point forecast series against truth series."""
    merged = forecast_df[["date", "p_ign"]].merge(truth_df, on="date", how="inner")
    if merged.empty:
        return {
            "n_days": 0,
            "threshold": float(threshold),
            "precision": float("nan"),
            "recall": float("nan"),
            "f1": float("nan"),
            "csi": float("nan"),
            "gss": float("nan"),
            "observed_fire_days": 0,
        }

    y_true = merged["observed_ignition"].astype(int).to_numpy()
    y_prob = merged["p_ign"].astype(float).to_numpy()
    y_pred = (y_prob >= threshold).astype(int)

    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    return {
        "n_days": int(len(merged)),
        "threshold": float(threshold),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "csi": float(compute_csi(y_true, y_prob, threshold=threshold)),
        "gss": float(compute_gilbert_skill_score(y_true, y_prob, threshold=threshold)),
        "observed_fire_days": int(y_true.sum()),
    }
