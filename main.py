"""Streamlit app: forecast future sea surface temperature from past values.

    streamlit run main.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from forecast import MODEL_FILE, Forecaster
from indicators import (coral_health, get_region, species_table, sst_category,
                        water_quality_index)
from sst_data import GRID_FILE

HISTORY_MONTHS = 36   # how much past data to show on the chart


@st.cache_resource
def load_forecaster():
    return Forecaster.load()


def plot_forecast(result):
    months = result["history_months"][-HISTORY_MONTHS:]
    history = result["history_sst"][-HISTORY_MONTHS:]
    labels = months + result["months"]
    x_hist = np.arange(len(months))
    x_fut = np.arange(len(months), len(labels))

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(x_hist, history, marker="o", markersize=3, color="#495057", label="Observed")
    ax.plot(x_fut, result["sst"], marker="o", color="#0B7285", linewidth=2.4, label="Forecast")
    ax.plot(x_fut, result["normal"], linestyle="--", color="#ADB5BD", label="Usual for that month")
    errors = result["typical_error"]
    if all(e is not None for e in errors):
        sst, err = np.array(result["sst"]), np.array(errors)
        ax.fill_between(x_fut, sst - err, sst + err, color="#0B7285", alpha=0.18,
                        label="Typical error (±1 RMSE)")
    step = max(1, len(labels) // 12)
    ax.set_xticks(np.arange(len(labels))[::step])
    ax.set_xticklabels(labels[::step], rotation=45, ha="right")
    ax.set_ylabel("SST (°C)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def forecast_tab(fc):
    col1, col2, col3 = st.columns(3)
    lat = col1.number_input("Latitude", -88.0, 88.0, 16.0, step=2.0)
    lon = col2.number_input("Longitude (-180 to 180, east is positive)", -180.0, 180.0, 66.0, step=2.0)
    lead = col3.slider("Months ahead", 1, fc.max_lead, 3)

    result = fc.forecast_point(lat, lon)
    if result["distance_deg"] > 6:
        st.warning("That point is far inland. Pick a location in or near the sea.")
        return
    if result["distance_deg"] > 1.5:
        st.info("That point is on land or between grid cells; using the nearest ocean cell.")

    shown_lon = result["cell_lon"] if result["cell_lon"] <= 180 else result["cell_lon"] - 360
    st.caption(f"Grid cell {result['cell_lat']:.0f}°, {shown_lon:.0f}° · "
               f"{get_region(result['cell_lat'], result['cell_lon'])} · "
               f"last observed month {fc.last_month}")

    i = lead - 1
    sst, normal, error = result["sst"][i], result["normal"][i], result["typical_error"][i]
    m1, m2, m3 = st.columns(3)
    m1.metric(f"Forecast for {result['months'][i]}", f"{sst:.2f} °C")
    m2.metric("Compared with usual", f"{sst - normal:+.2f} °C")
    if error is not None:
        m3.metric("Typical error at this lead", f"±{error:.2f} °C")

    fig = plot_forecast(result)
    st.pyplot(fig)
    plt.close(fig)

    with st.expander("Forecast table"):
        st.dataframe(pd.DataFrame({
            "Month": result["months"],
            "Forecast SST (°C)": np.round(result["sst"], 2),
            "Usual SST (°C)": np.round(result["normal"], 2),
            "Difference (°C)": np.round(np.array(result["sst"]) - np.array(result["normal"]), 2),
        }), use_container_width=True, hide_index=True)

    st.markdown(f"### Temperature-based indicators for {result['months'][i]}")
    st.caption("Rules of thumb based only on the forecast temperature. "
               "They are not measurements of water chemistry.")
    score, label = water_quality_index(sst)
    a, b, c = st.columns(3)
    a.metric("Temperature category", sst_category(sst))
    b.metric("Coral", coral_health(sst))
    c.metric("Thermal comfort index", f"{score}/100 ({label})")
    st.dataframe(pd.DataFrame(species_table(sst)), use_container_width=True, hide_index=True)


def map_tab(fc):
    col1, col2 = st.columns(2)
    lead = col1.slider("Months ahead", 1, fc.max_lead, 3, key="map_lead")
    view = col2.radio("Show", ["Temperature", "Difference from usual"], horizontal=True)

    sst, anomaly = fc.forecast_map(lead)
    fig, ax = plt.subplots(figsize=(11, 5))
    if view == "Temperature":
        mesh = ax.pcolormesh(fc.data.lon, fc.data.lat, sst, cmap="RdYlBu_r", shading="nearest")
        label = "SST (°C)"
    else:
        limit = float(np.nanpercentile(np.abs(anomaly), 99))
        mesh = ax.pcolormesh(fc.data.lon, fc.data.lat, anomaly, cmap="RdBu_r",
                             vmin=-limit, vmax=limit, shading="nearest")
        label = "Difference from usual (°C)"
    fig.colorbar(mesh, ax=ax, label=label)
    ax.set_facecolor("#DEE2E6")   # land
    ax.set_xlabel("Longitude (°E)")
    ax.set_ylabel("Latitude")
    ax.set_title(f"Forecast for {fc.target_label(lead)}")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def accuracy_tab(fc):
    metrics = fc.metrics
    if not metrics:
        st.info("No metrics saved with this model.")
        return
    st.markdown(
        f"Tested on forecasts for **{metrics['test_period_start']} onwards**, which the model "
        "did not see during training. Baselines: *Climatology* assumes the usual temperature "
        "for that month; *Persistence* assumes the current difference from usual stays the same.")
    rows = [{"Model": name, "RMSE (°C)": round(r["rmse"], 3), "MAE (°C)": round(r["mae"], 3)}
            for name, r in sorted(metrics["results"].items(), key=lambda kv: kv[1]["rmse"])]
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    plot = Path("plots") / "rmse_by_lead.png"
    if plot.exists():
        st.image(str(plot))
    if metrics.get("missing_months"):
        st.caption("Months with no data file (interpolated): " + ", ".join(metrics["missing_months"]))


def main():
    st.set_page_config(page_title="Sea Surface Temperature Forecast", page_icon="🌊", layout="wide")
    st.title("🌊 Sea Surface Temperature Forecast")
    st.markdown("Forecasts future monthly sea surface temperature from past values (NOAA ERSST v6).")

    if not Path(GRID_FILE).exists() or not Path(MODEL_FILE).exists():
        st.error("Model or data not found. Run `python train_model.py` first "
                 "(and `python prepare_data.py` if `sst_grid.npz` is missing).")
        st.stop()

    fc = load_forecaster()
    st.caption(f"Model: {fc.model_name} · data through {fc.last_month} · "
               f"forecasts up to {fc.max_lead} months ahead")

    tab1, tab2, tab3 = st.tabs(["Forecast", "Map", "Accuracy"])
    with tab1:
        forecast_tab(fc)
    with tab2:
        map_tab(fc)
    with tab3:
        accuracy_tab(fc)


if __name__ == "__main__":
    main()
