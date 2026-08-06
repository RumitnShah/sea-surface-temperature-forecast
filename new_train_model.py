# =========================
# IMPORT LIBRARIES
# =========================

import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor


# =========================
# 1. LOAD DATASET
# =========================

df = pd.read_csv("sst_combined.csv")

df.rename(columns={
    "lat": "latitude",
    "lon": "longitude"
}, inplace=True)

df["ssta"] = 0


# =========================
# 2. DATA PREPROCESSING
# =========================

df["time"] = pd.to_datetime(df["time"], errors="coerce")
df = df.dropna(subset=["time", "sst"])

df["quarter"]     = df["time"].dt.quarter
df["day_of_year"] = df["time"].dt.dayofyear

df["month_sin"] = df["sin_month"]
df["month_cos"] = df["cos_month"]

df["doy_sin"] = np.sin(2 * np.pi * df["day_of_year"] / 365)
df["doy_cos"] = np.cos(2 * np.pi * df["day_of_year"] / 365)


# =========================
# 🔥 2.5 DATA VISUALIZATION (NEW)
# =========================

print("\nGenerating visualizations...")

# 1. SST Distribution
plt.figure()
df["sst"].hist(bins=40)
plt.title("SST Distribution")
plt.xlabel("Temperature (°C)")
plt.ylabel("Frequency")
plt.savefig("sst_distribution.png")
plt.close()

# 2. SST vs Latitude
plt.figure()
plt.scatter(df["latitude"], df["sst"], alpha=0.2)
plt.title("SST vs Latitude")
plt.xlabel("Latitude")
plt.ylabel("SST")
plt.savefig("sst_vs_latitude.png")
plt.close()

# 3. Monthly Average SST
monthly_avg = df.groupby("month")["sst"].mean()
plt.figure()
monthly_avg.plot(marker='o')
plt.title("Average SST by Month")
plt.xlabel("Month")
plt.ylabel("SST")
plt.savefig("sst_monthly.png")
plt.close()

# 4. Yearly Trend
yearly_avg = df.groupby("year")["sst"].mean()
plt.figure()
yearly_avg.plot(marker='o')
plt.title("Yearly SST Trend")
plt.xlabel("Year")
plt.ylabel("SST")
plt.savefig("sst_yearly.png")
plt.close()

# 5. Correlation Heatmap
plt.figure(figsize=(10,8))
sns.heatmap(df.corr(numeric_only=True), cmap="coolwarm")
plt.title("Feature Correlation Heatmap")
plt.savefig("correlation_heatmap.png")
plt.close()

print("Visualizations saved as PNG files")


# =========================
# 3. FEATURE ENGINEERING
# =========================

df["lat_grid"] = (df["latitude"] / 2.0).round() * 2.0
df["lon_grid"] = (df["longitude"] / 2.0).round() * 2.0

df["lat_x_month"] = df["latitude"] * df["month"]
df["lon_x_month"] = df["longitude"] * df["month"]

df["lat_squared"] = df["latitude"] ** 2
df["ssta_abs"] = df["ssta"].abs()

df = df[(df["sst"] > -2) & (df["sst"] < 40)]

print(f"Dataset shape: {df.shape}")


# =========================
# 4. FEATURE SELECTION
# =========================

FEATURE_COLS = [
    "latitude", "longitude",
    "lat_grid", "lon_grid",
    "ssta", "ssta_abs",
    "year", "quarter",
    "month_sin", "month_cos",
    "doy_sin", "doy_cos",
    "lat_x_month", "lon_x_month",
    "lat_squared"
]

X = df[FEATURE_COLS]
y = df["sst"]


# =========================
# 5. DATA SCALING
# =========================

scaler_X = StandardScaler()
X_scaled = scaler_X.fit_transform(X)


# =========================
# 6. TRAIN-TEST SPLIT
# =========================

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y.values,
    test_size=0.2,
    shuffle=True,
    random_state=42
)


# =========================
# 7. MODEL DEFINITIONS
# =========================

models = {
    "Random Forest": RandomForestRegressor(
        n_estimators=300,
        max_depth=20,
        min_samples_split=10,
        min_samples_leaf=5,
        max_features=0.6,
        n_jobs=-1,
        random_state=42
    ),

    "XGBoost": XGBRegressor(
        n_estimators=600,
        learning_rate=0.03,
        max_depth=8,
        subsample=0.7,
        colsample_bytree=0.7,
        colsample_bylevel=0.7,
        reg_alpha=0.3,
        reg_lambda=2.0,
        min_child_weight=10,
        gamma=0.05,
        random_state=42,
        verbosity=0,
        n_jobs=-1
    )
}


# =========================
# 8. TRAINING & EVALUATION
# =========================

results = {}

for name, model in models.items():
    print(f"\n🔹 Training: {name}")

    model.fit(X_train, y_train)

    train_preds = model.predict(X_train)
    test_preds  = model.predict(X_test)

    train_rmse = np.sqrt(mean_squared_error(y_train, train_preds))
    test_rmse  = np.sqrt(mean_squared_error(y_test, test_preds))
    test_r2    = r2_score(y_test, test_preds)

    print(f"Train RMSE : {train_rmse:.3f}")
    print(f"Test RMSE  : {test_rmse:.3f}")
    print(f"R² Score   : {test_r2:.4f}")

    if test_rmse > train_rmse * 1.5:
        print("⚠️ Overfitting detected")
    elif train_rmse > 3:
        print("⚠️ Underfitting detected")
    else:
        print("✅ Good model fit")

    results[name] = {
        "model": model,
        "test_rmse": test_rmse
    }

best_name = min(results, key=lambda x: results[x]["test_rmse"])
best_model = results[best_name]["model"]

# =========================
# 🔥 8.5 MODEL VISUALIZATION (NEW)
# =========================

print("\nGenerating model performance visualizations...")

model_names = []
rmse_values = []

# Collect RMSE values
for name in results:
    model_names.append(name)
    rmse_values.append(results[name]["test_rmse"])

# -------------------------
# 1. RMSE Comparison Bar Plot
# -------------------------
plt.figure()
plt.bar(model_names, rmse_values)
plt.title("Model Comparison (Test RMSE)")
plt.xlabel("Model")
plt.ylabel("RMSE")
plt.savefig("model_comparison.png")
plt.close()

# -------------------------
# 2. Actual vs Predicted Plot (Best Model)
# -------------------------
best_preds = best_model.predict(X_test)

plt.figure()
plt.scatter(y_test, best_preds, alpha=0.3)
plt.xlabel("Actual SST")
plt.ylabel("Predicted SST")
plt.title("Actual vs Predicted SST")
plt.savefig("actual_vs_predicted.png")
plt.close()

# -------------------------
# 3. Residual Plot
# -------------------------
residuals = y_test - best_preds

plt.figure()
plt.scatter(best_preds, residuals, alpha=0.3)
plt.xlabel("Predicted SST")
plt.ylabel("Residuals")
plt.title("Residual Plot")
plt.axhline(0)
plt.savefig("residual_plot.png")
plt.close()

print("Model visualizations saved!")

# =========================
# 9. SELECT BEST MODEL
# =========================

best_name = min(results, key=lambda x: results[x]["test_rmse"])
best_model = results[best_name]["model"]

print(f"\n🏆 Best Model: {best_name}")


# =========================
# 10. SAVE MODEL
# =========================

pickle.dump(best_model, open("best_model.pkl", "wb"))
pickle.dump(scaler_X, open("scaler_X.pkl", "wb"))

with open("feature_cols.txt", "w") as f:
    f.write("\n".join(FEATURE_COLS))

print("\n🎉 Training complete!")