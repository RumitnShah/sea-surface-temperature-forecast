# =========================
# IMPORT LIBRARIES
# =========================

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from artifacts import read_model_name

# =========================
# PAGE CONFIGURATION
# =========================

def configure_page():
    st.set_page_config(
        page_title="Marine Water Quality Prediction System",
        page_icon="🌊",
        layout="wide"
    )
    st.title("🌊 Marine Water Quality Prediction System")
    st.markdown("Predict Sea Surface Temperature (SST) and analyze marine ecosystem & water quality.")

# =========================
# LOAD MODEL
# =========================

@st.cache_resource
def load_artifacts():
    model = pickle.load(open("best_model.pkl", "rb"))
    scaler = pickle.load(open("scaler_X.pkl", "rb"))
    name = read_model_name(model)
    return model, scaler, name

# =========================
# FEATURE FUNCTION
# =========================

def build_features(lat, lon, year, month, doy):
    quarter = int(np.ceil(month / 3))

    lat_grid = round(lat / 2.0) * 2.0
    lon_grid = round(lon / 2.0) * 2.0

    return np.array([[  
        lat, lon, lat_grid, lon_grid,
        0, 0,  # placeholder (ssta removed but kept for model compatibility)
        year, quarter,
        np.sin(2*np.pi*month/12),
        np.cos(2*np.pi*month/12),
        np.sin(2*np.pi*doy/365),
        np.cos(2*np.pi*doy/365),
        lat * month, lon * month,
        lat ** 2
    ]])

# =========================
# REGION FUNCTION
# =========================

def get_region(lat, lon):
    if lon > 180:
        lon -= 360

    if 5 <= lat <= 25 and 55 <= lon <= 75:
        return "🌊 Arabian Sea"
    if 5 <= lat <= 25 and 80 <= lon <= 100:
        return "🌊 Bay of Bengal"
    if -40 <= lat <= 30 and 20 <= lon <= 120:
        return "🌊 Indian Ocean"
    if -60 <= lat <= 60 and (lon <= -100 or lon >= 120):
        return "🌊 Pacific Ocean"
    if -60 <= lat <= 60 and -70 <= lon <= 20:
        return "🌊 Atlantic Ocean"
    if 30 <= lat <= 45 and -5 <= lon <= 40:
        return "🌊 Mediterranean Sea"
    if 0 <= lat <= 25 and 100 <= lon <= 125:
        return "🌊 South China Sea"
    if 10 <= lat <= 25 and -85 <= lon <= -60:
        return "🌊 Caribbean Sea"
    if lat < -50:
        return "❄️ Southern Ocean (Antarctica)"
    if lat > 65:
        return "❄️ Arctic Ocean"

    return "🌍 Open Ocean"

# =========================
# BASIC SST SAFETY (OLD FEATURE)
# =========================

def sst_safety(sst):
    if sst < 10:
        return "❄️ Cold"
    elif sst < 20:
        return "🌤️ Moderate"
    elif sst < 28:
        return "🌊 Warm"
    elif sst < 32:
        return "⚠️ High"
    else:
        return "🔥 Extreme"

# =========================
# PREDICTION VISUALIZATION
# =========================

def monthly_prediction_series(model, scaler, lat, lon, year):
    predictions = []

    for month in range(1, 13):
        doy = int(pd.Timestamp(year=int(year), month=month, day=15).day_of_year)
        features = build_features(lat, lon, year, month, doy)
        scaled = scaler.transform(features)
        predictions.append(float(model.predict(scaled)[0]))

    return pd.DataFrame({
        "Month": list(range(1, 13)),
        "Predicted SST": predictions
    })

def plot_temperature_band(sst):
    bands = [
        (-2, 10, "Cold", "#4FC3F7"),
        (10, 20, "Moderate", "#81C784"),
        (20, 28, "Warm", "#FFB74D"),
        (28, 32, "High", "#FB8C00"),
        (32, 40, "Extreme", "#E53935"),
    ]

    fig, ax = plt.subplots(figsize=(9, 2.2))

    for start, end, label, color in bands:
        ax.axvspan(start, end, color=color, alpha=0.32)
        ax.text((start + end) / 2, 0.58, label, ha="center", va="center",
                fontsize=9, fontweight="bold")

    ax.axvline(sst, color="#111111", linewidth=2.5)
    ax.scatter([sst], [0.25], color="#111111", s=70, zorder=3)
    ax.text(sst, 0.08, f"{sst:.2f} °C", ha="center", va="center",
            fontsize=10, fontweight="bold")

    ax.set_xlim(-2, 40)
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.set_xlabel("Predicted Sea Surface Temperature (°C)")
    ax.set_title("Predicted Temperature Risk Band")
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    return fig

def plot_monthly_prediction(monthly_predictions):
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(
        monthly_predictions["Month"],
        monthly_predictions["Predicted SST"],
        marker="o",
        linewidth=2.4,
        color="#0B7285"
    )
    ax.set_xticks(range(1, 13))
    ax.set_xlabel("Month")
    ax.set_ylabel("Predicted SST (°C)")
    ax.set_title("Monthly Predicted Sea Surface Temperature")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig

# =========================
# NEW WATER QUALITY FUNCTIONS
# =========================

def water_quality_index(sst):
    if 20 <= sst <= 28:
        return 90, "Excellent"
    elif 15 <= sst <= 32:
        return 70, "Good"
    elif 10 <= sst <= 35:
        return 50, "Moderate"
    else:
        return 30, "Poor"

def habitat_quality(sst):
    if 22 <= sst <= 28:
        return "🌿 Highly Suitable"
    elif 15 <= sst <= 32:
        return "⚠️ Moderately Suitable"
    else:
        return "❌ Unsuitable"

def coral_health(sst):
    if sst > 30:
        return "🔥 Coral Bleaching Risk"
    elif sst > 28:
        return "⚠️ Stress"
    else:
        return "✅ Healthy"

def fish_survival(sst):
    if 18 <= sst <= 26:
        return "🐟 Optimal"
    elif 10 <= sst <= 30:
        return "⚠️ Survival Possible"
    else:
        return "❌ Risky"

def environmental_risk(sst):
    if sst > 30:
        return "🔥 High Risk"
    elif sst > 27:
        return "⚠️ Medium Risk"
    else:
        return "✅ Low Risk"

# =========================
# PREDICTION TAB
# =========================

def prediction_tab(model, scaler):

    col1, col2 = st.columns(2)

    with col1:
        lat = st.slider("Latitude", -90.0, 90.0, 20.0)
        lon = st.slider("Longitude", 0.0, 360.0, 70.0)

        year = st.number_input("Year", 2000, 2100, 2024)
        month = st.selectbox("Month", list(range(1, 13)))

    if st.button("🔮 Predict SST"):

        doy = int(pd.Timestamp(year=year, month=month, day=15).day_of_year)

        features = build_features(lat, lon, year, month, doy)
        scaled = scaler.transform(features)
        sst = model.predict(scaled)[0]

        # =========================
        # OUTPUT
        # =========================

        st.metric("🌡 SST", f"{sst:.2f} °C")

        st.markdown("### 📈 Prediction Graphs")
        band_fig = plot_temperature_band(sst)
        st.pyplot(band_fig)
        plt.close(band_fig)

        monthly_predictions = monthly_prediction_series(model, scaler, lat, lon, year)
        monthly_fig = plot_monthly_prediction(monthly_predictions)
        st.pyplot(monthly_fig)
        plt.close(monthly_fig)

        region = get_region(lat, lon)
        st.success(f"📍 Location: {region}")

        st.info(f"🌊 SST Category: {sst_safety(sst)}")

        # =========================
        # WATER QUALITY
        # =========================

        st.markdown("### 🌊 Water Quality Analysis")

        score, label = water_quality_index(sst)

        st.metric("💧 Water Quality Index", f"{score}/100")
        st.progress(score / 100)

        st.write(f"Status: {label}")
        st.write(f"🐠 Habitat: {habitat_quality(sst)}")
        st.write(f"🪸 Coral Health: {coral_health(sst)}")
        st.write(f"🐟 Fish Survival: {fish_survival(sst)}")
        st.write(f"🌪 Environmental Risk: {environmental_risk(sst)}")

        # =========================
        # MARINE SPECIES TABLE
        # =========================

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
                "Temp Range": f"{mn}-{mx}°C",
                "Status": status_sp
            })

        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

# =========================
# HEATMAP TAB (OLD FEATURE KEPT)
# =========================

def heatmap_tab(model, scaler):

    st.subheader("🗺️ SST Heatmap")

    lat_min = st.number_input("Lat Min", -90.0, 90.0, -10.0)
    lat_max = st.number_input("Lat Max", -90.0, 90.0, 30.0)

    lon_min = st.number_input("Lon Min", 0.0, 360.0, 50.0)
    lon_max = st.number_input("Lon Max", 0.0, 360.0, 90.0)

    resolution = st.selectbox("Resolution", [1.0, 2.0, 5.0], index=1)

    if st.button("Generate Heatmap"):

        lats = np.arange(lat_min, lat_max, resolution)
        lons = np.arange(lon_min, lon_max, resolution)

        year = 2024
        month = 6
        doy = int(pd.Timestamp(year=year, month=month, day=15).day_of_year)

        Z = []
        for la in lats:
            row = []
            for lo in lons:
                f = build_features(la, lo, year, month, doy)
                row.append(model.predict(scaler.transform(f))[0])
            Z.append(row)

        Z = np.array(Z)

        fig, ax = plt.subplots(figsize=(10, 6))
        c = ax.contourf(lons, lats, Z, cmap="RdYlBu_r")
        plt.colorbar(c, label="SST (°C)")

        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_title("SST Heatmap")

        st.pyplot(fig)

# =========================
# MAIN
# =========================

def main():
    configure_page()

    model, scaler, model_name = load_artifacts()
    st.success(f"✅ Loaded model: {model_name}")

    tab1, tab2 = st.tabs(["🔍 Prediction", "🗺️ Heatmap"])

    with tab1:
        prediction_tab(model, scaler)

    with tab2:
        heatmap_tab(model, scaler)

    st.markdown("----")
    st.markdown("<center>Marine Water Quality Prediction System</center>", unsafe_allow_html=True)

# =========================
# RUN
# =========================

if __name__ == "__main__":
    main()
