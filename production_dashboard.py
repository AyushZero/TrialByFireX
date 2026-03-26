import streamlit as st
import numpy as np
import pandas as pd
import folium
from streamlit_folium import st_folium
import json, os, yaml, joblib

# Setup
ROOT = os.path.dirname(os.path.abspath(__file__))

st.set_page_config(
    page_title="Production Wildfire Inference",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Load config and norm params
@st.cache_resource
def load_assets():
    with open(os.path.join(ROOT, "config.yaml")) as f:
        cfg = yaml.safe_load(f)
    with open(os.path.join(ROOT, "data", "processed", "norm_params.json")) as f:
        norm = json.load(f)
    model = joblib.load(os.path.join(ROOT, "models", "physics_logistic.pkl"))
    return cfg, norm, model

cfg, norm_params, model = load_assets()

def normalize(val, var_name):
    # Safety fallback if var not found
    if var_name not in norm_params:
        return 0.0
    vmin = norm_params[var_name]["min"]
    vmax = norm_params[var_name]["max"]
    return (val - vmin) / (vmax - vmin if vmax != vmin else 1.0)

# Sidebar Inputs
st.sidebar.title("🔥 Operational Inputs")
st.sidebar.markdown("Adjust atmospheric & geographic parameters to simulate ignition probability.")

# Atmospheric
st.sidebar.subheader("Atmospheric Conditions")
t_max = st.sidebar.slider("Max Temperature (°C)", 0.0, 50.0, 30.0)
rh_min = st.sidebar.slider("Min Relative Humidity (%)", 5.0, 100.0, 15.0)
u10_max = st.sidebar.slider("Max Wind Speed (m/s)", 0.0, 30.0, 8.0)
sm_top = st.sidebar.slider("Soil Moisture (fraction)", 0.0, 1.0, 0.1)

# Geographic / Vegetative
st.sidebar.subheader("Geospatial Context")
ndvi = st.sidebar.slider("NDVI (Greenness)", -0.2, 1.0, 0.6)
ndwi = st.sidebar.slider("NDWI (Canopy Water)", -0.5, 0.5, -0.2)
slope = st.sidebar.slider("Terrain Slope (°)", 0.0, 45.0, 15.0)

# Historical
st.sidebar.subheader("Local Activity (7-Day)")
frp_hist = st.sidebar.number_input("Trailing FRP (MW)", 0.0, 5000.0, 0.0)
count_hist = st.sidebar.number_input("Trailing Ignitions (N)", 0, 100, 0)

# Run Physics Formulations
t_norm = normalize(t_max, "t_max")
rh_norm = normalize(rh_min, "rh_min")
u_norm = normalize(u10_max, "u10_max")
sm_norm = normalize(sm_top, "sm_top")
ndvi_norm = normalize(ndvi, "ndvi")
ndwi_norm = normalize(ndwi, "ndwi")
slope_norm = normalize(slope, "slope")
frp_norm = normalize(frp_hist, "frp_hist")
c_norm = normalize(count_hist, "count_hist")

F_avail = max(0, ndvi_norm)
F_dry = (cfg["alpha"]["alpha1"] * (1 - ndwi_norm) +
         cfg["alpha"]["alpha2"] * (1 - rh_norm) +
         cfg["alpha"]["alpha3"] * (1 - sm_norm))
G_spread = 1 + (cfg["beta"]["beta1"] * u_norm) + (cfg["beta"]["beta2"] * slope_norm)
H_hist = (cfg["gamma"]["gamma1"] * frp_norm) + (cfg["gamma"]["gamma2"] * c_norm)

R_phys = (F_avail * F_dry * G_spread) + H_hist
p_ign = model.predict_proba(np.array([[R_phys]]))[0][1]

# Main UI
st.title("Physics-Guided Ignition Risk Simulator")
st.markdown("---")

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Current R_phys", f"{R_phys:.2f}")
col2.metric("Fuel Availability", f"{F_avail:.2f}")
col3.metric("Fuel Dryness", f"{F_dry:.2f}")
col4.metric("Spread Potential", f"{G_spread:.2f}")
col5.metric("Local History (Biased)", f"{H_hist:.2f}")

st.markdown("### Estimated Ignition Probability")
# Colored alert bar depending on threshold
if p_ign > 0.8:
    st.error(f"🛑 CRITICAL RISK: {p_ign*100:.2f}% probability of secondary ignition.")
elif p_ign > 0.4:
    st.warning(f"⚠ ELEVATED RISK: {p_ign*100:.2f}% probability of ignition.")
else:
    st.success(f"✅ NOMINAL CONDITIONS: {p_ign*100:.2f}% probability of ignition.")
st.progress(float(p_ign))

# Map
st.markdown("### Simulated Operations Map")
st.markdown("Displaying a simulated high-risk focal point in Northern California.")

focal_lat = 39.87  # Dixie Fire origin approx
focal_lon = -121.38

m = folium.Map(location=[focal_lat, focal_lon], zoom_start=9, tiles="CartoDB dark_matter")
color = "red" if p_ign > 0.4 else "green"

folium.CircleMarker(
    location=[focal_lat, focal_lon],
    radius=15,
    popup=f"Simualted Probability: {p_ign:.2f}",
    color=color,
    fill=True,
    fill_color=color,
    fill_opacity=0.6 * p_ign
).add_to(m)

st_data = st_folium(m, width=900, height=450)
