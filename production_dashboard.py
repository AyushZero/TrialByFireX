"""
🔥 TrialsByFireX – Production Inference Dashboard
Dark-theme Streamlit dashboard for wildfire risk prediction.

Users input weather conditions → get real-time ignition probability.
Features:
  • Interactive risk calculator with physics sliders
  • Dark heatmap over California grid
  • Fire risk alerts and zone highlighting
  • Model comparison panel

Launch: streamlit run production_dashboard.py
"""

import os, sys, yaml, pickle
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

# ── Page config ──────────────────────────────────────────────────
st.set_page_config(
    page_title="TrialsByFireX – Wildfire Risk Predictor",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Dark theme CSS ───────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

    * { font-family: 'Inter', sans-serif; }

    .main { background-color: #0a0a0f; }
    .stApp { background-color: #0a0a0f; }
    [data-testid="stSidebar"] { background-color: #0f0f1a; border-right: 1px solid #1a1a2e; }
    [data-testid="stHeader"] { background-color: #0a0a0f; }

    h1 { color: #ff6b35; font-weight: 700; }
    h2 { color: #f0a500; font-weight: 600; }
    h3 { color: #e0e0e0; font-weight: 600; }
    p, label, span, .stMarkdown { color: #c0c0c0; }

    .risk-box {
        padding: 20px; border-radius: 12px;
        text-align: center; font-size: 2.5rem; font-weight: 700;
        margin: 10px 0;
    }
    .risk-low { background: linear-gradient(135deg, #1b5e20, #2e7d32); color: #a5d6a7; border: 2px solid #43a047; }
    .risk-med { background: linear-gradient(135deg, #e65100, #f57c00); color: #ffe0b2; border: 2px solid #ff9800; }
    .risk-high { background: linear-gradient(135deg, #b71c1c, #d32f2f); color: #ffcdd2; border: 2px solid #f44336; }

    .metric-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        padding: 16px; border-radius: 10px; border: 1px solid #2a2a4e;
        text-align: center; margin: 5px;
    }
    .metric-label { font-size: 0.8rem; color: #888; text-transform: uppercase; }
    .metric-value { font-size: 1.6rem; font-weight: 700; color: #f0a500; }

    div[data-testid="stMetricValue"] { color: #f0a500; }
    div[data-testid="stMetricLabel"] { color: #888; }

    .stSlider > div > div > div > div { background-color: #ff6b35; }
    .stSelectbox > div { background-color: #1a1a2e; }

    /* Tab styling */
    .stTabs [data-baseweb="tab"] { color: #888; }
    .stTabs [aria-selected="true"] { color: #ff6b35; border-bottom-color: #ff6b35; }
</style>
""", unsafe_allow_html=True)


# ── Data loading ─────────────────────────────────────────────────
@st.cache_data
def load_config():
    with open(os.path.join(ROOT, "config.yaml")) as f:
        return yaml.safe_load(f)

@st.cache_resource
def load_model(name):
    path = os.path.join(ROOT, "models", f"{name}.pkl")
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    return None

@st.cache_data
def load_features():
    import xarray as xr
    path = os.path.join(ROOT, "data", "features", "features.nc")
    if os.path.exists(path):
        ds = xr.open_dataset(path)
        return ds
    return None

@st.cache_data
def load_comparison():
    path = os.path.join(ROOT, "outputs", "model_comparison.csv")
    if os.path.exists(path):
        return pd.read_csv(path, index_col=0)
    return None


# ── Load resources ───────────────────────────────────────────────
cfg = load_config()
model_phys = load_model("physics_logistic")


# ── Sidebar ──────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("# 🔥 TrialsByFireX")
    st.markdown("**Physics-Guided Wildfire<br>Risk Prediction**",
                unsafe_allow_html=True)
    st.markdown("---")

    st.markdown("### 📊 Model Info")
    if model_phys:
        a = model_phys.coef_[0, 0]
        b = model_phys.intercept_[0]
        st.code(f"p = σ({a:.3f}·R + {b:.3f})", language="text")
    else:
        st.warning("No trained model found")

    st.markdown("### ⏱ Training Period")
    st.markdown(f"{cfg['time']['start']} → {cfg['time']['end']}")

    st.markdown("### 🔬 Physics Weights")
    st.markdown(f"""
    **α** (dryness): {cfg['alpha']}
    **β** (spread): {cfg['beta']}
    **γ** (history): {cfg['gamma']}
    """)

    st.markdown("---")
    st.caption("Conference-ready research pipeline")
    st.caption("ERA5 · MODIS · SRTM · FIRMS")


# ── Main content tabs ────────────────────────────────────────────
tab_predict, tab_map, tab_results, tab_about = st.tabs([
    "🧮 Predict Risk", "🗺️ Risk Map", "📊 Results", "📐 About"
])


# ════════════════════════════════════════════════════════════════
# TAB 1: PREDICT RISK (interactive calculator)
# ════════════════════════════════════════════════════════════════
with tab_predict:
    st.markdown("# 🧮 Real-Time Ignition Risk Calculator")
    st.markdown("*Adjust the weather and terrain conditions below to compute "
                "the physics-guided ignition probability.*")
    st.markdown("---")

    col_input, col_result = st.columns([2, 1])

    with col_input:
        st.markdown("### 🌡️ Meteorological Conditions")
        c1, c2, c3 = st.columns(3)
        with c1:
            temp_C = st.slider("Temperature (°C)", 0, 50, 35, 1,
                               help="Daily maximum 2m temperature")
            rh_pct = st.slider("Relative Humidity (%)", 0, 100, 20, 1,
                               help="Daily minimum relative humidity")
        with c2:
            wind_ms = st.slider("Wind Speed (m/s)", 0, 30, 8, 1,
                                help="Daily maximum 10m wind speed")
            precip_mm = st.slider("Precipitation (mm)", 0.0, 50.0, 0.0, 0.5,
                                  help="Daily total precipitation")
        with c3:
            sm = st.slider("Soil Moisture (m³/m³)", 0.0, 0.5, 0.15, 0.01,
                           help="Top-layer volumetric soil moisture")

        st.markdown("### 🌿 Vegetation & Terrain")
        c4, c5, c6 = st.columns(3)
        with c4:
            ndvi_val = st.slider("NDVI", 0.0, 1.0, 0.45, 0.01,
                                 help="Normalized vegetation index")
        with c5:
            ndwi_val = st.slider("NDWI", 0.0, 1.0, 0.3, 0.01,
                                 help="Normalized water index")
        with c6:
            slope_deg = st.slider("Slope (degrees)", 0, 45, 10, 1,
                                  help="Terrain slope")

        st.markdown("### 🔥 Fire History (last 7 days)")
        c7, c8 = st.columns(2)
        with c7:
            frp_hist = st.slider("FRP History (MW)", 0.0, 100.0, 0.0, 1.0,
                                 help="Decayed fire radiative power nearby")
        with c8:
            count_hist = st.slider("Fire Count", 0, 20, 0, 1,
                                   help="Decayed hotspot count nearby")

    # ── Normalise inputs ──
    T_norm = np.clip(temp_C / 50.0, 0, 1)
    RH_norm = np.clip(rh_pct / 100.0, 0, 1)
    U_norm = np.clip(wind_ms / 30.0, 0, 1)
    SM_norm = np.clip(sm / 0.5, 0, 1)
    NDVI_norm = ndvi_val
    NDWI_norm = ndwi_val
    slope_norm = np.clip(slope_deg / 45.0, 0, 1)
    FRP_norm = np.clip(frp_hist / 100.0, 0, 1)
    Count_norm = np.clip(count_hist / 20.0, 0, 1)

    alpha = cfg["alpha"]
    beta = cfg["beta"]
    gamma = cfg["gamma"]

    F_avail = NDVI_norm
    F_dry = alpha["alpha1"]*(1-NDWI_norm) + alpha["alpha2"]*(1-RH_norm) + alpha["alpha3"]*(1-SM_norm)
    G_spread = 1 + beta["beta1"]*U_norm + beta["beta2"]*slope_norm
    H_history = gamma["gamma1"]*FRP_norm + gamma["gamma2"]*Count_norm
    R_phys = T_norm * F_avail * F_dry * G_spread + H_history

    if model_phys:
        a = model_phys.coef_[0, 0]
        b = model_phys.intercept_[0]
        p_ign = float(1 / (1 + np.exp(-(a * R_phys + b))))
    else:
        p_ign = float(1 / (1 + np.exp(-R_phys)))

    with col_result:
        st.markdown("### ⚡ Prediction")

        # Risk level with giant colored box
        if p_ign < 0.3:
            risk_class, risk_label = "risk-low", "LOW RISK"
        elif p_ign < 0.6:
            risk_class, risk_label = "risk-med", "MODERATE RISK"
        else:
            risk_class, risk_label = "risk-high", "HIGH RISK"

        st.markdown(f"""
        <div class="risk-box {risk_class}">
            {risk_label}<br>
            <span style="font-size:3.5rem">{p_ign:.1%}</span>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")

        # Component breakdown
        st.markdown("#### Component Breakdown")
        components = {
            "R_phys": R_phys,
            "F_avail (fuel)": F_avail,
            "F_dry (dryness)": F_dry,
            "G_spread (wind+slope)": G_spread,
            "H_history (fire hist)": H_history,
        }
        for name, val in components.items():
            bar_pct = min(val / 2.0 * 100, 100)  # Scale for display
            st.markdown(f"**{name}**: `{val:.4f}`")
            st.progress(min(float(val), 1.0))

        # Risk gauge
        gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=p_ign * 100,
            title={"text": "Ignition Probability (%)"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#ff6b35"},
                "steps": [
                    {"range": [0, 30], "color": "#1b5e20"},
                    {"range": [30, 60], "color": "#e65100"},
                    {"range": [60, 100], "color": "#b71c1c"},
                ],
                "threshold": {
                    "line": {"color": "white", "width": 3},
                    "thickness": 0.8,
                    "value": p_ign * 100,
                },
            },
        ))
        gauge.update_layout(
            height=250, margin=dict(l=20, r=20, t=40, b=20),
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#c0c0c0"),
        )
        st.plotly_chart(gauge, use_container_width=True)


# ════════════════════════════════════════════════════════════════
# TAB 2: RISK MAP (dark map with dark tiles)
# ════════════════════════════════════════════════════════════════
with tab_map:
    st.markdown("# 🗺️ California Fire Risk Map")

    ds = load_features()
    if ds is not None and model_phys is not None:
        from src.grid import grid_info
        gi = grid_info(cfg)

        dates = pd.to_datetime(ds.time.values)
        c_date, c_thresh = st.columns([1, 1])
        with c_date:
            selected_date = st.date_input(
                "📅 Select Date",
                value=pd.Timestamp("2023-07-15").date(),
                min_value=dates[0].date(), max_value=dates[-1].date(),
            )
        with c_thresh:
            threshold = st.slider("⚠ Alert Threshold", 0.0, 1.0, 0.5, 0.05)

        target = np.datetime64(str(selected_date))
        t_idx = np.argmin(np.abs(ds.time.values - target))

        if "R_phys" in ds:
            R_day = ds["R_phys"].isel(time=t_idx).values
            a = model_phys.coef_[0, 0]
            b = model_phys.intercept_[0]
            prob_grid = 1 / (1 + np.exp(-(a * R_day + b)))

            # Dark-themed heatmap
            fig = go.Figure()

            # Add heatmap
            fig.add_trace(go.Heatmap(
                z=prob_grid,
                x=gi["lons"], y=gi["lats"],
                colorscale=[
                    [0.0, "#0a0a0f"],
                    [0.15, "#1a3300"],
                    [0.3, "#2d6a00"],
                    [0.45, "#8a8a00"],
                    [0.6, "#cc6600"],
                    [0.75, "#cc3300"],
                    [0.9, "#990000"],
                    [1.0, "#ff0000"],
                ],
                zmin=0, zmax=1,
                colorbar=dict(
                    title="p_ign",
                    titlefont=dict(color="#c0c0c0"),
                    tickfont=dict(color="#c0c0c0"),
                    bgcolor="rgba(0,0,0,0.5)",
                ),
                hovertemplate="Lat: %{y:.2f}°N<br>Lon: %{x:.2f}°W<br>p_ign: %{z:.4f}<extra></extra>",
            ))

            # Dark map layout
            fig.update_layout(
                title=dict(
                    text=f"Ignition Probability — {selected_date}",
                    font=dict(color="#e0e0e0", size=18),
                ),
                xaxis=dict(
                    title="Longitude", color="#888",
                    gridcolor="#1a1a2e", zerolinecolor="#1a1a2e",
                ),
                yaxis=dict(
                    title="Latitude", color="#888",
                    gridcolor="#1a1a2e", zerolinecolor="#1a1a2e",
                ),
                plot_bgcolor="#0a0a0f",
                paper_bgcolor="#0a0a0f",
                height=650,
                font=dict(color="#c0c0c0"),
            )

            # State outline (approximate California border)
            ca_lats = [32.5, 34.5, 36.0, 37.0, 38.0, 39.5, 40.5, 42.0,
                       42.0, 41.0, 39.5, 38.5, 37.5, 36.5, 35.5, 34.0, 32.5]
            ca_lons = [-117.1, -117.5, -118.0, -119.0, -120.5, -122.0, -122.5, -124.0,
                       -120.0, -120.0, -120.0, -121.5, -122.0, -121.5, -120.5, -118.5, -117.1]

            fig.add_trace(go.Scatter(
                x=ca_lons, y=ca_lats,
                mode="lines",
                line=dict(color="#444", width=2),
                showlegend=False,
                hoverinfo="skip",
            ))

            st.plotly_chart(fig, use_container_width=True)

            # Statistics row
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Mean p_ign", f"{np.nanmean(prob_grid):.4f}")
            c2.metric("Max p_ign", f"{np.nanmax(prob_grid):.4f}")
            c3.metric("Cells > threshold",
                       f"{np.sum(prob_grid > threshold):,} / {prob_grid.size:,}")
            c4.metric("R_phys mean", f"{np.nanmean(R_day):.4f}")

            if np.any(prob_grid > threshold):
                st.error(f"⚠ **ALERT**: {np.sum(prob_grid > threshold)} cells "
                         f"exceed the {threshold:.0%} threshold!")
    else:
        st.warning("Run `python run_preprocess.py --synthetic` and "
                   "`python run_train.py` first.")


# ════════════════════════════════════════════════════════════════
# TAB 3: RESULTS (model comparison + analysis outputs)
# ════════════════════════════════════════════════════════════════
with tab_results:
    st.markdown("# 📊 Model Comparison & Analysis")

    comp_df = load_comparison()
    if comp_df is not None:
        # Metric highlights
        c1, c2, c3 = st.columns(3)
        best_roc = comp_df["auc_roc"].idxmax()
        best_brier = comp_df["brier_score"].idxmin()

        c1.metric("🏆 Best AUC-ROC", best_roc,
                   f"{comp_df.loc[best_roc, 'auc_roc']:.4f}")
        c2.metric("🏆 Best Brier", best_brier,
                   f"{comp_df.loc[best_brier, 'brier_score']:.4f}")
        if "log_loss" in comp_df.columns:
            best_ll = comp_df["log_loss"].idxmin()
            c3.metric("🏆 Best Log-Loss", best_ll,
                       f"{comp_df.loc[best_ll, 'log_loss']:.4f}")

        st.dataframe(comp_df.style.format("{:.4f}"), use_container_width=True)

        st.markdown("---")

        # Show analysis images in tabs
        analysis_tabs = st.tabs([
            "ROC/PR", "Confusion", "SHAP", "Ablation",
            "Threshold", "Seasonal", "Correlation"
        ])

        outs = os.path.join(ROOT, "outputs")

        with analysis_tabs[0]:
            c1, c2 = st.columns(2)
            for f, c in [("roc_curves.png", c1), ("pr_curves.png", c2)]:
                p = os.path.join(outs, f)
                if os.path.exists(p): c.image(p)

        with analysis_tabs[1]:
            cm_files = [f for f in os.listdir(outs) if f.startswith("confusion_")]
            if cm_files:
                cols = st.columns(min(3, len(cm_files)))
                for i, f in enumerate(sorted(cm_files)):
                    cols[i % 3].image(os.path.join(outs, f),
                        caption=f.replace("confusion_", "").replace(".png", "").replace("_", " ").title())

        with analysis_tabs[2]:
            c1, c2 = st.columns(2)
            for f, c in [("shap_random_forest.png", c1), ("shap_xgboost.png", c2)]:
                p = os.path.join(outs, f)
                if os.path.exists(p): c.image(p)

        with analysis_tabs[3]:
            p = os.path.join(outs, "ablation_study.png")
            if os.path.exists(p):
                st.image(p, caption="Ablation Study — Factor Contribution")
            abl_csv = os.path.join(outs, "ablation_results.csv")
            if os.path.exists(abl_csv):
                st.dataframe(pd.read_csv(abl_csv, index_col=0).style.format("{:.4f}"))

        with analysis_tabs[4]:
            c1, c2 = st.columns(2)
            for f, c in [("threshold_analysis.png", c1), ("reliability_diagram.png", c2)]:
                p = os.path.join(outs, f)
                if os.path.exists(p): c.image(p)

        with analysis_tabs[5]:
            p = os.path.join(outs, "seasonal_analysis.png")
            if os.path.exists(p): st.image(p)

        with analysis_tabs[6]:
            p = os.path.join(outs, "correlation_heatmap.png")
            if os.path.exists(p): st.image(p)

    else:
        st.warning("Run `python run_train.py` first.")


# ════════════════════════════════════════════════════════════════
# TAB 4: ABOUT (physics formulas)
# ════════════════════════════════════════════════════════════════
with tab_about:
    st.markdown("# 📐 Physics-Guided Risk Index")

    st.markdown("""
    ### Composite Factor Equations

    | Factor | Formula | Physical Interpretation |
    |--------|---------|------------------------|
    | **F_avail** | NDVĪ(t) | Live fuel biomass available to burn |
    | **F_dry** | α₁(1−NDWĪ) + α₂(1−RH̄) + α₃(1−SM̄) | Canopy + atmospheric + soil dryness |
    | **G_spread** | 1 + β₁·Ū + β₂·θ̄ | Wind and slope amplification |
    | **H_history** | γ₁·FRP̄ₕ + γ₂·Count̄ₕ | Recent fire persistence |

    ### Risk Index
    > **R_phys(t) = T̄(t) · F_avail · F_dry · G_spread + H_history**

    ### Calibrated Probability
    > **p_ign(t) = σ(a · R_phys + b) = 1 / (1 + exp(-(a · R_phys + b)))**

    ### Data Sources
    | Source | Variables | Resolution |
    |--------|-----------|------------|
    | ERA5 | Temperature, RH, wind, precipitation, soil moisture | 0.25°, hourly |
    | MODIS | NDVI, NDWI | 1km, 16-day |
    | SRTM | DEM → slope | 30m |
    | FIRMS | Fire hotspots (FRP, type, confidence) | 375m, daily |

    ### Study Region
    **California, USA** (32°N–42°N, 124°W–114°W), 2021–2023
    """)


# ── Footer ───────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<center style='color:#555'>"
    "TrialsByFireX · Physics-Guided Ignition Probability Modelling · "
    "ERA5 · MODIS · SRTM · FIRMS"
    "</center>",
    unsafe_allow_html=True,
)
