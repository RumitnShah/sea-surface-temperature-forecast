import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import os

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Marine Water Quality Prediction System",
    page_icon="🌊",
    layout="wide"
)

st.title("🌊 Marine Water Quality Prediction System")
st.markdown("Predict Sea Surface Temperature (SST) and analyze marine impact.")

# =========================
# LOAD MODEL
# =========================
@st.cache_resource
def load_artifacts():
    model   = pickle.load(open("best_model.pkl", "rb"))
    scaler  = pickle.load(open("scaler_X.pkl", "rb"))
    name    = open("best_model_name.txt").read().strip()
    return model, scaler, name

model, scaler, model_name = load_artifacts()
st.success(f"✅ Loaded model: {model_name}")

# =========================
# FEATURE FUNCTION
# =========================
def build_features(lat, lon, year, month, doy, ssta):
    quarter = int(np.ceil(month / 3))
    lat_grid = round(lat / 2.0) * 2.0
    lon_grid = round(lon / 2.0) * 2.0

    return np.array([[
        lat, lon, lat_grid, lon_grid,
        ssta, abs(ssta),
        year, quarter,
        np.sin(2*np.pi*month/12),
        np.cos(2*np.pi*month/12),
        np.sin(2*np.pi*doy/365),
        np.cos(2*np.pi*doy/365),
        lat*month, lon*month,
        lat**2
    ]])

# =========================
# REGION FUNCTION
# =========================
def get_region(lat, lon):
    if -30 <= lat <= 30:
        if 40 <= lon <= 100:
            return "🌊 Indian Ocean (Tropical)"
        elif -160 <= lon <= -70:
            return "🌊 Pacific Ocean (Tropical)"
        elif -70 <= lon <= 20:
            return "🌊 Atlantic Ocean (Tropical)"
        else:
            return "🌍 Equatorial Waters"
    elif lat > 30:
        return "❄️ Northern Hemisphere"
    else:
        return "❄️ Southern Hemisphere"

# =========================
# SAFETY FUNCTION
# =========================
def sst_safety(sst):
    if sst < 10:
        return "❄️ Cold", "#4FC3F7", "Polar conditions"
    elif sst < 20:
        return "🌤️ Moderate", "#81C784", "Stable waters"
    elif sst < 28:
        return "🌊 Warm", "#FFB74D", "Optimal marine life"
    elif sst < 32:
        return "⚠️ High", "#FB8C00", "Coral stress"
    else:
        return "🔥 Extreme", "#E53935", "Marine heatwave"

# =========================
# TABS
# =========================
tab1, tab2 = st.tabs(["🔍 Prediction", "🗺️ Heatmap"])

# =========================
# TAB 1 — PREDICTION
# =========================
with tab1:

    col1, col2 = st.columns(2)

    with col1:
        lat = st.slider("Latitude", -90.0, 90.0, 20.0)
        lon = st.slider("Longitude", 0.0, 360.0, 70.0)

        year = st.number_input("Year", 0, 9999, 2024)
        month = st.selectbox("Month", list(range(1,13)))

    with col2:
        ssta = st.number_input("SSTA (°C)", -5.0, 5.0, 0.0)

    if st.button("🔮 Predict SST"):

        doy = int(pd.Timestamp(year=year, month=month, day=15).day_of_year)

        features = build_features(lat, lon, year, month, doy, ssta)
        scaled = scaler.transform(features)
        sst = model.predict(scaled)[0]

        st.metric("🌡 SST", f"{sst:.2f} °C")

        # Region
        region = get_region(lat, lon)
        st.info(f"📍 Region: {region}")

        # Safety
        status, color, desc = sst_safety(sst)

        st.markdown(f"""
        <div style='background:{color}22;border-left:6px solid {color};
                    padding:1rem;border-radius:8px'>
        <h3 style='color:{color}'>{status}</h3>
        <p>{desc}</p>
        </div>
        """, unsafe_allow_html=True)

        # Species
        st.markdown("### 🐟 Marine Life Impact")

        species = [
            ("🪸 Coral Reef", 20, 28),
            ("🐟 Tuna", 5, 30),
            ("🦐 Shrimp", 10, 30),
            ("🐢 Sea Turtle", 20, 32),
            ("🌿 Seagrass", 15, 30),
            ("🐠 Reef Fish", 22, 28),
            ("🦈 Shark", 8, 26),
            ("🐋 Whale", 2, 20),
            ("🐧 Penguin", -2, 15),
            ("🦑 Squid", 5, 25),
            ("🦀 Crab", 5, 25),
        ]

        rows = []
        for name, mn, mx in species:
            status_sp = "✅ Suitable" if mn <= sst <= mx else "⚠️ At Risk"
            rows.append({
                "Species": name,
                "Range": f"{mn}-{mx}°C",
                "Status": status_sp
            })

        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

# =========================
# TAB 2 — HEATMAP
# =========================
with tab2:

    st.subheader("🗺️ SST Heatmap")

    col1, col2 = st.columns(2)

    with col1:
        lat_min = st.number_input("Lat Min", -90.0, 90.0, -10.0)
        lat_max = st.number_input("Lat Max", -90.0, 90.0, 30.0)

    with col2:
        lon_min = st.number_input("Lon Min", 0.0, 360.0, 50.0)
        lon_max = st.number_input("Lon Max", 0.0, 360.0, 90.0)

    resolution = st.selectbox("Resolution", [1.0, 2.0, 5.0], index=1)

    if st.button("Generate Heatmap"):

        lats = np.arange(lat_min, lat_max, resolution)
        lons = np.arange(lon_min, lon_max, resolution)

        doy = int(pd.Timestamp(year=year, month=month, day=15).day_of_year)

        Z = []
        for la in lats:
            row = []
            for lo in lons:
                f = build_features(la, lo, year, month, doy, ssta)
                row.append(model.predict(scaler.transform(f))[0])
            Z.append(row)

        Z = np.array(Z)

        fig, ax = plt.subplots(figsize=(10,6))
        c = ax.contourf(lons, lats, Z, cmap="RdYlBu_r")
        plt.colorbar(c, label="SST (°C)")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_title("SST Heatmap")

        st.pyplot(fig)

# =========================
# FOOTER
# =========================
st.markdown("---")
st.markdown("<center>Marine Water Quality Prediction System</center>", unsafe_allow_html=True)