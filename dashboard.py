"""
🔥 TrialsByFireX - Wildfire Ignition Probability Dashboard
Streamlit interactive dashboard for the physics-guided ignition model.

Launch: streamlit run dashboard.py
"""

import os, sys, yaml, pickle
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

# -- Page config --------------------------------------------------
st.set_page_config(
    page_title="TrialsByFireX - Wildfire Ignition Probability",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -- Custom CSS ---------------------------------------------------
st.markdown("""
<style>
    .main { background-color: #0e1117; }
    .stMetric { background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
                padding: 15px; border-radius: 10px; border: 1px solid #333; }
    h1 { color: #ff6b35; }
    h2 { color: #f7c59f; }
    .sidebar .sidebar-content { background: #1a1a2e; }
    div[data-testid="stMetricValue"] { font-size: 1.8rem; }
</style>
""", unsafe_allow_html=True)


# -- Data loading -------------------------------------------------
@st.cache_data
def load_config():
    with open(os.path.join(ROOT, "config.yaml")) as f:
        return yaml.safe_load(f)

@st.cache_data
def load_features():
    import xarray as xr
    path = os.path.join(ROOT, "data", "features", "features.nc")
    if not os.path.exists(path):
        return None
    return xr.open_dataset(path)

@st.cache_data
def load_comparison():
    path = os.path.join(ROOT, "outputs", "model_comparison.csv")
    if os.path.exists(path):
        return pd.read_csv(path, index_col=0)
    return None

@st.cache_resource
def load_model(name):
    path = os.path.join(ROOT, "models", f"{name}.pkl")
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    return None


# -- Sidebar ------------------------------------------------------
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/fire-element.png", width=60)
    st.title("🔥 TrialsByFireX")
    st.markdown("**Physics-Guided Ignition Probability**")
    st.markdown("---")

    cfg = load_config()
    st.subheader("📍 Study Region")
    st.write(f"**{cfg['region']['name']}**")
    st.write(f"Lat: {cfg['region']['lat_min']} deg - {cfg['region']['lat_max']} deg")
    st.write(f"Lon: {cfg['region']['lon_min']} deg - {cfg['region']['lon_max']} deg")

    st.subheader("⏱ Time Range")
    st.write(f"{cfg['time']['start']} [RIGHT] {cfg['time']['end']}")

    st.subheader("[CONFIG] Physics Weights")
    st.json({
        "α (dryness)": cfg["alpha"],
        "β (spread)": cfg["beta"],
        "γ (history)": cfg["gamma"],
    })

    st.markdown("---")
    st.caption("Built for college research project")


# -- Main content -------------------------------------------------
st.title("🔥 Wildfire Ignition Probability Dashboard")
st.markdown("**Physics-Guided Modelling using ERA5, MODIS, SRTM & FIRMS**")

# -- Tab layout ---------------------------------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Model Comparison",
    "🗺️ Probability Map",
    "📈 Feature Analysis",
    "🔬 Diagnostics",
    "📐 Physics Index",
])

# ════════════════════════════════════════════════════════════════
# TAB 1: Model Comparison
# ════════════════════════════════════════════════════════════════
with tab1:
    comp_df = load_comparison()
    if comp_df is not None:
        st.header("Model Performance Comparison")

        # Metric cards
        col1, col2, col3 = st.columns(3)
        best_roc = comp_df["auc_roc"].idxmax()
        best_pr = comp_df["auc_pr"].idxmax()
        best_brier = comp_df["brier_score"].idxmin()

        col1.metric("🏆 Best AUC-ROC", best_roc,
                     f"{comp_df.loc[best_roc, 'auc_roc']:.4f}")
        col2.metric("🏆 Best AUC-PR", best_pr,
                     f"{comp_df.loc[best_pr, 'auc_pr']:.4f}")
        col3.metric("🏆 Best Brier", best_brier,
                     f"{comp_df.loc[best_brier, 'brier_score']:.4f}")

        st.markdown("---")

        # Bar chart
        fig = make_subplots(rows=1, cols=3,
                            subplot_titles=["AUC-ROC [UP]", "AUC-PR [UP]", "Brier Score [DOWN]"])

        colors = px.colors.qualitative.Set2[:len(comp_df)]

        for i, (metric, higher_better) in enumerate([
            ("auc_roc", True), ("auc_pr", True), ("brier_score", False)
        ]):
            fig.add_trace(
                go.Bar(
                    x=comp_df.index, y=comp_df[metric],
                    marker_color=colors,
                    text=comp_df[metric].round(4),
                    textposition="outside",
                    showlegend=False,
                ),
                row=1, col=i+1,
            )

        fig.update_layout(height=400, template="plotly_dark",
                          title_text="Model Comparison - Test Set")
        st.plotly_chart(fig, use_container_width=True)

        # Table
        st.subheader("📋 Detailed Metrics")
        st.dataframe(
            comp_df.style.highlight_max(subset=["auc_roc", "auc_pr"], color="#2e7d32")
                         .highlight_min(subset=["brier_score"], color="#2e7d32")
                         .format("{:.4f}"),
            use_container_width=True,
        )

        # Show saved plots
        st.markdown("---")
        st.subheader("📊 ROC & PR Curves")
        c1, c2 = st.columns(2)
        roc_path = os.path.join(ROOT, "outputs", "roc_curves.png")
        pr_path = os.path.join(ROOT, "outputs", "pr_curves.png")
        if os.path.exists(roc_path):
            c1.image(roc_path, caption="ROC Curves")
        if os.path.exists(pr_path):
            c2.image(pr_path, caption="Precision-Recall Curves")

    else:
        st.warning("[WARN] No model comparison data found. Run `python run_train.py` first.")


# ════════════════════════════════════════════════════════════════
# TAB 2: Probability Map
# ════════════════════════════════════════════════════════════════
with tab2:
    st.header("🗺️ Ignition Probability Map")

    ds = load_features()
    model_phys = load_model("physics_logistic")

    if ds is not None and model_phys is not None:
        from src.grid import grid_info
        gi = grid_info(cfg)

        # Date selector
        dates = pd.to_datetime(ds.time.values)
        min_date = dates[0].date()
        max_date = dates[-1].date()

        col_date, col_thresh = st.columns(2)
        with col_date:
            selected_date = st.date_input(
                "Select date", value=pd.Timestamp("2023-07-15").date(),
                min_value=min_date, max_value=max_date,
            )
        with col_thresh:
            threshold = st.slider("Alert threshold", 0.0, 1.0, 0.5, 0.05)

        # Find nearest time
        target = np.datetime64(str(selected_date))
        t_idx = np.argmin(np.abs(ds.time.values - target))

        # Get R_phys for that day
        if "R_phys" in ds:
            R_day = ds["R_phys"].isel(time=t_idx).values
            a = model_phys.coef_[0, 0]
            b = model_phys.intercept_[0]
            prob_grid = 1 / (1 + np.exp(-(a * R_day + b)))

            # Plotly heatmap
            fig = go.Figure(data=go.Heatmap(
                z=prob_grid,
                x=gi["lons"],
                y=gi["lats"],
                colorscale=[[0, "#2d6a4f"], [0.25, "#95d5b2"],
                            [0.5, "#ffff3f"], [0.75, "#ff9f1c"], [1, "#e63946"]],
                zmin=0, zmax=1,
                colorbar=dict(title="p_ign"),
            ))
            fig.update_layout(
                title=f"Ignition Probability — {selected_date}",
                xaxis_title="Longitude",
                yaxis_title="Latitude",
                height=600,
                template="plotly_dark",
            )
            st.plotly_chart(fig, use_container_width=True)

            # Statistics
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Mean p_ign", f"{np.nanmean(prob_grid):.4f}")
            c2.metric("Max p_ign", f"{np.nanmax(prob_grid):.4f}")
            c3.metric("Cells > threshold",
                       f"{np.sum(prob_grid > threshold):,}")
            c4.metric("R_phys mean", f"{np.nanmean(R_day):.4f}")

            # High-risk cells
            if np.any(prob_grid > threshold):
                st.warning(f"[WARN] {np.sum(prob_grid > threshold)} cells above "
                           f"threshold ({threshold})")
    else:
        st.warning("[WARN] Run `python run_preprocess.py --synthetic` and "
                   "`python run_train.py` first.")


# ════════════════════════════════════════════════════════════════
# TAB 3: Feature Analysis
# ════════════════════════════════════════════════════════════════
with tab3:
    st.header("📈 Feature Analysis")

    c1, c2 = st.columns(2)

    # Correlation heatmap
    corr_path = os.path.join(ROOT, "outputs", "correlation_heatmap.png")
    if os.path.exists(corr_path):
        c1.image(corr_path, caption="Feature Correlation Matrix")

    # Feature distributions
    dist_path = os.path.join(ROOT, "outputs", "feature_distributions.png")
    if os.path.exists(dist_path):
        c2.image(dist_path, caption="Feature Distributions by Label")

    # SHAP values
    st.markdown("---")
    st.subheader("🔍 SHAP Feature Importance")

    c3, c4 = st.columns(2)
    shap_rf_path = os.path.join(ROOT, "outputs", "shap_random_forest.png")
    shap_xgb_path = os.path.join(ROOT, "outputs", "shap_xgboost.png")

    if os.path.exists(shap_rf_path):
        c3.image(shap_rf_path, caption="Random Forest SHAP")
    if os.path.exists(shap_xgb_path):
        c4.image(shap_xgb_path, caption="XGBoost SHAP")

    if not os.path.exists(shap_rf_path) and not os.path.exists(shap_xgb_path):
        st.info("SHAP plots will appear after running `python run_train.py`")


# ════════════════════════════════════════════════════════════════
# TAB 4: Diagnostics
# ════════════════════════════════════════════════════════════════
with tab4:
    st.header("🔬 Model Diagnostics")

    # Confusion matrices
    st.subheader("Confusion Matrices")
    cm_files = [f for f in os.listdir(os.path.join(ROOT, "outputs"))
                if f.startswith("confusion_") and f.endswith(".png")]
    if cm_files:
        cols = st.columns(min(3, len(cm_files)))
        for i, f in enumerate(sorted(cm_files)):
            cols[i % 3].image(
                os.path.join(ROOT, "outputs", f),
                caption=f.replace("confusion_", "").replace(".png", "").replace("_", " ").title(),
            )

    # Reliability diagram
    st.markdown("---")
    st.subheader("Calibration")
    c1, c2 = st.columns(2)

    rel_path = os.path.join(ROOT, "outputs", "reliability_diagram.png")
    thresh_path = os.path.join(ROOT, "outputs", "threshold_analysis.png")

    if os.path.exists(rel_path):
        c1.image(rel_path, caption="Reliability Diagram")
    if os.path.exists(thresh_path):
        c2.image(thresh_path, caption="Threshold Analysis (F1 vs Threshold)")

    # Seasonal analysis
    st.markdown("---")
    st.subheader("Seasonal Performance")
    seasonal_path = os.path.join(ROOT, "outputs", "seasonal_analysis.png")
    if os.path.exists(seasonal_path):
        st.image(seasonal_path, caption="Monthly / Seasonal AUC-ROC Breakdown")

    # Stratified performance
    st.markdown("---")
    st.subheader("Stratified Performance")
    c1, c2 = st.columns(2)
    slope_csv = os.path.join(ROOT, "outputs", "performance_by_slope.csv")
    ndvi_csv = os.path.join(ROOT, "outputs", "performance_by_ndvi.csv")
    if os.path.exists(slope_csv):
        c1.markdown("**By Slope**")
        c1.dataframe(pd.read_csv(slope_csv))
    if os.path.exists(ndvi_csv):
        c2.markdown("**By NDVI**")
        c2.dataframe(pd.read_csv(ndvi_csv))


# ════════════════════════════════════════════════════════════════
# TAB 5: Physics Index
# ════════════════════════════════════════════════════════════════
with tab5:
    st.header("📐 Physics-Guided Risk Index")

    st.markdown("""
    ### Composite Factors

    | Factor | Formula | Physical Meaning |
    |--------|---------|------------------|
    | **F_avail** | NDVI̅(t) | Live fuel biomass |
    | **F_dry** | α₁(1−NDWI̅) + α₂(1−RH̅) + α₃(1−SM̅) | Fuel dryness |
    | **G_spread** | 1 + β₁.U̅ + β₂.θ̅ | Wind-slope spread |
    | **H_history** | γ₁.FRP̅ₕᵢₛₜ + γ₂.Count̅ₕᵢₛₜ | Recent fire activity |

    ### Risk Index
    **R_phys(t) = T̅(t) . F_avail . F_dry . G_spread + H_history**

    ### Probability
    **p_ign(t) = σ(a . R_phys + b)**
    """)

    # Interactive physics calculator
    st.markdown("---")
    st.subheader("🧮 Interactive Calculator")

    col1, col2, col3 = st.columns(3)
    with col1:
        T_norm = st.slider("T̃ (temperature)", 0.0, 1.0, 0.5, 0.05)
        ndvi_norm = st.slider("NDVĨ", 0.0, 1.0, 0.5, 0.05)
        ndwi_norm = st.slider("NDWĨ", 0.0, 1.0, 0.5, 0.05)
    with col2:
        rh_norm = st.slider("RH̃ (humidity)", 0.0, 1.0, 0.5, 0.05)
        sm_norm = st.slider("SM̃ (soil moisture)", 0.0, 1.0, 0.5, 0.05)
        u_norm = st.slider("Ũ (wind speed)", 0.0, 1.0, 0.3, 0.05)
    with col3:
        slope_norm = st.slider("θ̃ (slope)", 0.0, 1.0, 0.2, 0.05)
        frp_norm = st.slider("FRP̃_hist", 0.0, 1.0, 0.0, 0.05)
        count_norm = st.slider("Count̃_hist", 0.0, 1.0, 0.0, 0.05)

    alpha = cfg["alpha"]
    beta = cfg["beta"]
    gamma = cfg["gamma"]

    F_avail = ndvi_norm
    F_dry = alpha["alpha1"]*(1-ndwi_norm) + alpha["alpha2"]*(1-rh_norm) + alpha["alpha3"]*(1-sm_norm)
    G_spread = 1 + beta["beta1"]*u_norm + beta["beta2"]*slope_norm
    H_history = gamma["gamma1"]*frp_norm + gamma["gamma2"]*count_norm
    R_phys = T_norm * F_avail * F_dry * G_spread + H_history

    # Apply model if available
    model_phys = load_model("physics_logistic")
    if model_phys is not None:
        a = model_phys.coef_[0, 0]
        b = model_phys.intercept_[0]
        p_ign = 1 / (1 + np.exp(-(a * R_phys + b)))
    else:
        p_ign = 1 / (1 + np.exp(-R_phys))  # default sigmoid

    # Display results
    st.markdown("---")
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("F_avail", f"{F_avail:.3f}")
    c2.metric("F_dry", f"{F_dry:.3f}")
    c3.metric("G_spread", f"{G_spread:.3f}")
    c4.metric("H_history", f"{H_history:.3f}")
    c5.metric("R_phys", f"{R_phys:.4f}")

    # Color-code probability
    if p_ign < 0.3:
        c6.metric("🟢 p_ign", f"{p_ign:.4f}")
    elif p_ign < 0.6:
        c6.metric("🟡 p_ign", f"{p_ign:.4f}")
    else:
        c6.metric("🔴 p_ign", f"{p_ign:.4f}")

    # R_phys timeseries
    ts_path = os.path.join(ROOT, "outputs", "R_phys_timeseries.png")
    if os.path.exists(ts_path):
        st.markdown("---")
        st.subheader("R_phys Temporal Evolution")
        st.image(ts_path, caption="Spatial-mean R_phys vs fire rate over test period")


# -- Footer -------------------------------------------------------
st.markdown("---")
st.markdown(
    "<center><small>TrialsByFireX — Physics-Guided Ignition Probability Modelling "
    "| Built with Streamlit</small></center>",
    unsafe_allow_html=True,
)
