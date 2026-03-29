"""
Black-theme single-day ignition map dashboard (data-driven).

Run:
    streamlit run single_day_map_dashboard.py
"""

import io
import json
import os
import sys

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yaml
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from src.grid import grid_info
from src.inference import run_inference_from_daily_dataframe


st.set_page_config(page_title="TrialsByFireX Daily Ignition Map", layout="wide")

st.markdown(
    """
<style>
    .stApp { background: #03060a; color: #e6edf3; }
    [data-testid="stSidebar"] { background: #0a0f14; border-right: 1px solid #1f2937; }
    [data-testid="stHeader"] { background: #03060a; }
    h1, h2, h3 { color: #f5f7fa; }
    .stMarkdown, .stText, label, p, span, div { color: #d0d7de; }
</style>
""",
    unsafe_allow_html=True,
)


EXPECTED_FEATURES = [
    "t_max", "rh_min", "u10_max", "sm_top", "ndvi", "ndwi", "slope", "frp_hist", "count_hist",
]


@st.cache_data
def load_config():
    with open(os.path.join(ROOT, "config.yaml"), "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


@st.cache_data
def load_daily_demo_file(filename):
    path = os.path.join(ROOT, "data", "demo", filename)
    if not os.path.exists(path):
        return pd.DataFrame(), path
    return pd.read_csv(path), path


@st.cache_data
def load_optimized_profile():
    p = os.path.join(ROOT, "outputs", "optimized_formula_profile.json")
    if not os.path.exists(p):
        return None
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def _build_grid_text(prob_grid, decimals=2):
    text = np.empty_like(prob_grid, dtype=object)
    for i in range(prob_grid.shape[0]):
        for j in range(prob_grid.shape[1]):
            v = prob_grid[i, j]
            text[i, j] = "" if not np.isfinite(v) else f"{v:.{decimals}f}"
    return text


def _stretch_bounds(values, q_low=0.05, q_high=0.95):
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 0.0, 1.0
    lo = float(np.quantile(finite, q_low))
    hi = float(np.quantile(finite, q_high))
    if hi <= lo:
        return float(np.nanmin(finite)), float(np.nanmax(finite))
    return lo, hi


def build_map(prob_grid, lats, lons, number_mode=True, scale_mode="P05-P95 stretch"):
    if scale_mode == "Absolute (0-1)":
        zmin, zmax = 0.0, 1.0
    elif scale_mode == "P01-P99 stretch":
        zmin, zmax = _stretch_bounds(prob_grid, 0.01, 0.99)
    else:
        zmin, zmax = _stretch_bounds(prob_grid, 0.05, 0.95)

    text = _build_grid_text(prob_grid, decimals=2)

    fig = go.Figure(
        data=[
            go.Heatmap(
                z=prob_grid,
                x=lons,
                y=lats,
                text=text,
                texttemplate="%{text}" if number_mode else None,
                textfont={"size": 9, "color": "#f8fafc"},
                colorscale=[
                    [0.0, "#0b132b"],
                    [0.2, "#1f3c88"],
                    [0.4, "#2a9d8f"],
                    [0.6, "#f4d35e"],
                    [0.8, "#ee964b"],
                    [1.0, "#f95738"],
                ],
                zmin=zmin,
                zmax=zmax,
                hovertemplate="Lat %{y:.2f}<br>Lon %{x:.2f}<br>p_ign %{z:.4f}<extra></extra>",
                colorbar={"title": "p_ign", "tickfont": {"color": "#e6edf3"}},
            )
        ]
    )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#03060a",
        plot_bgcolor="#03060a",
        xaxis={"title": "Longitude", "gridcolor": "#1f2937", "zerolinecolor": "#1f2937"},
        yaxis={
            "title": "Latitude",
            "gridcolor": "#1f2937",
            "zerolinecolor": "#1f2937",
            "scaleanchor": "x",
            "scaleratio": 1,
        },
        margin={"l": 10, "r": 10, "t": 50, "b": 10},
        height=780,
    )

    return fig, (zmin, zmax)


def compute_truth_metrics(day_df, prob_grid, cfg, threshold):
    if "ignition" not in day_df.columns:
        return None

    gi = grid_info(cfg)
    truth_grid = np.full_like(prob_grid, np.nan, dtype=float)

    lat = day_df["latitude"].to_numpy(dtype=float)
    lon = day_df["longitude"].to_numpy(dtype=float)
    ign = day_df["ignition"].to_numpy(dtype=float)

    lat_idx = ((lat - gi["lat_min"]) / gi["resolution"]).astype(int)
    lon_idx = ((lon - gi["lon_min"]) / gi["resolution"]).astype(int)

    valid = (
        np.isfinite(ign)
        & (lat_idx >= 0)
        & (lon_idx >= 0)
        & (lat_idx < gi["n_lat"])
        & (lon_idx < gi["n_lon"])
    )

    truth_grid[lat_idx[valid], lon_idx[valid]] = ign[valid]

    mask = np.isfinite(prob_grid) & np.isfinite(truth_grid)
    if mask.sum() == 0:
        return None

    y_true = truth_grid[mask].astype(int)
    y_prob = prob_grid[mask]
    y_pred = (y_prob >= threshold).astype(int)

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "n_fire_cells": int(y_true.sum()),
        "n_cells": int(mask.sum()),
    }


cfg = load_config()
gi = grid_info(cfg)
opt_payload = load_optimized_profile()

st.title("California Daily Ignition Probability")
st.caption("Provide one day grid data -> model runs -> probability map by grid cell.")

left, right = st.columns([1.3, 2.7])

with left:
    st.subheader("Input")
    data_mode = st.radio("Data source", ["Demo dataset", "Upload CSV"], index=0)

    day_df = pd.DataFrame()
    source_label = ""
    if data_mode == "Demo dataset":
        demo_file = st.selectbox(
            "Choose demo day",
            ["daily_grid_2022-07-15.csv", "daily_grid_2023-07-15.csv"],
            index=1,
        )
        day_df, source_path = load_daily_demo_file(demo_file)
        source_label = source_path
    else:
        uploaded = st.file_uploader("Upload daily grid CSV", type=["csv"])
        if uploaded is not None:
            day_df = pd.read_csv(io.BytesIO(uploaded.getvalue()))
            source_label = uploaded.name

    st.markdown("---")
    normalized_input = st.checkbox("Input features already normalized [0,1]", value=True)
    scale_mode = st.radio("Color scaling", ["P05-P95 stretch", "P01-P99 stretch", "Absolute (0-1)"], index=0)
    number_mode = st.checkbox("Show probability number in each grid", value=True)
    threshold = st.slider("Fire threshold for accuracy check", 0.0, 1.0, 0.55, 0.01)

    model_name = "physics_logistic_baseline_retrained"
    model_options = ["Baseline model"]
    if opt_payload is not None:
        model_options.append("Optimized model")
    model_choice = st.radio("Model", model_options, index=0)
    if model_choice == "Optimized model":
        model_name = "physics_logistic_optimized"

    run_now = st.button("Run model for this day", type="primary")

    if not day_df.empty:
        st.caption(f"Rows loaded: {len(day_df)}")
        st.caption(f"Source: {source_label}")

with right:
    if not run_now:
        st.warning("Load a demo/upload CSV and click 'Run model for this day'.")
    else:
        required = {"latitude", "longitude"}.union(EXPECTED_FEATURES)
        missing = sorted([c for c in required if c not in day_df.columns and c != "R_phys"])
        if day_df.empty:
            st.error("No data loaded. Choose a demo dataset or upload CSV.")
            st.stop()
        if missing:
            st.error("Missing required columns: " + ", ".join(missing))
            st.stop()

        try:
            da, meta = run_inference_from_daily_dataframe(
                day_df=day_df,
                cfg=cfg,
                model_dir=os.path.join(ROOT, cfg["paths"]["models"]),
                norm_params_path=os.path.join(ROOT, cfg["paths"]["processed_data"], "norm_params.json"),
                normalized_input=normalized_input,
                model_name=model_name,
            )
        except FileNotFoundError as e:
            st.error(
                "Selected model file not found. "
                "Run: c:/Minor/TrialsByFireX/.venv/Scripts/python.exe scripts/retrain_physics_profiles.py"
            )
            st.exception(e)
            st.stop()

        prob_grid = da.values
        fig, (zmin, zmax) = build_map(prob_grid, gi["lats"], gi["lons"], number_mode=number_mode, scale_mode=scale_mode)

        title_date = "unknown"
        if "date" in day_df.columns and day_df["date"].nunique() == 1:
            title_date = str(day_df["date"].iloc[0])

        fig.update_layout(title=f"Ignition Probability Grid - {title_date}")
        st.plotly_chart(fig, use_container_width=True)

        finite = prob_grid[np.isfinite(prob_grid)]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Mean", f"{np.nanmean(finite):.4f}" if finite.size else "nan")
        c2.metric("Median", f"{np.nanmedian(finite):.4f}" if finite.size else "nan")
        c3.metric("P95", f"{np.nanquantile(finite, 0.95):.4f}" if finite.size else "nan")
        c4.metric("Max", f"{np.nanmax(finite):.4f}" if finite.size else "nan")

        st.info(
            f"Rows used: {meta['rows_valid']}/{meta['rows_total']} | Dropped: {meta['rows_dropped']} | "
            f"Color range: [{zmin:.3f}, {zmax:.3f}]"
        )

        metrics = compute_truth_metrics(day_df, prob_grid, cfg, threshold)
        if metrics is not None:
            st.subheader("Truth Check (fire/no-fire)")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Accuracy", f"{metrics['accuracy']:.4f}")
            m2.metric("Precision", f"{metrics['precision']:.4f}")
            m3.metric("Recall", f"{metrics['recall']:.4f}")
            m4.metric("F1", f"{metrics['f1']:.4f}")
            st.caption(f"Fire cells in truth: {metrics['n_fire_cells']} / {metrics['n_cells']}")
        else:
            st.warning("No ignition truth column found in input; skipping fire/no-fire check.")

        csv_df = pd.DataFrame(
            [
                {"latitude": gi["lats"][i], "longitude": gi["lons"][j], "p_ign": float(prob_grid[i, j])}
                for i in range(prob_grid.shape[0])
                for j in range(prob_grid.shape[1])
                if np.isfinite(prob_grid[i, j])
            ]
        )
        st.download_button(
            "Download predicted grid CSV",
            data=csv_df.to_csv(index=False).encode("utf-8"),
            file_name="predicted_daily_grid.csv",
            mime="text/csv",
        )
