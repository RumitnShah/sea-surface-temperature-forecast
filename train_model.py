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

df = pd.read_csv(
    r"C:\Users\HP\OneDrive\Documents\GitHub\marine-water-quality-classification\nceiErsstv5_b0fe_5755_37cf.csv",   # ← update path if needed
    skiprows=[1]   # skips the units row (UTC, m, degrees_north, ...)
)

df.columns = ["time", "depth", "latitude", "longitude", "sst", "ssta"]

# =========================
# 2. PREPROCESSING
# =========================

df["time"] = pd.to_datetime(df["time"], errors="coerce")
df = df.dropna(subset=["time"])

# Drop rows where target (sst) is missing
df = df.dropna(subset=["sst"])

# Fill ssta (anomaly) with 0 if missing — anomaly of 0 means no deviation
df["ssta"] = df["ssta"].fillna(0)

# Drop depth — constant 0.0 across all rows, adds no signal
df = df.drop(columns=["depth"])

# Time features
df["month"]       = df["time"].dt.month
df["year"]        = df["time"].dt.year
df["quarter"]     = df["time"].dt.quarter
df["day_of_year"] = df["time"].dt.dayofyear

# Cyclical time encoding — so Dec and Jan are "close" to the model
df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
df["doy_sin"]   = np.sin(2 * np.pi * df["day_of_year"] / 365)
df["doy_cos"]   = np.cos(2 * np.pi * df["day_of_year"] / 365)

# Location grid — round to 2° bins to help model learn regional patterns
df["lat_grid"] = (df["latitude"]  / 2.0).round() * 2.0
df["lon_grid"] = (df["longitude"] / 2.0).round() * 2.0

# Physics-informed features
df["lat_x_month"]  = df["latitude"]  * df["month"]     # seasonal-latitude interaction
df["lon_x_month"]  = df["longitude"] * df["month"]     # seasonal-longitude interaction
df["lat_squared"]  = df["latitude"]  ** 2              # non-linear lat effect
df["ssta_abs"]     = df["ssta"].abs()                  # magnitude of anomaly

# Filter realistic SST range (°C)
df = df[(df["sst"] > -2) & (df["sst"] < 40)]

print(f"Dataset shape after cleaning: {df.shape}")
print(f"SST range: {df['sst'].min():.2f} → {df['sst'].max():.2f} °C")
print(f"Date range: {df['time'].min().date()} → {df['time'].max().date()}")

# =========================
# 3. VISUALIZATION
# =========================

print("\nGenerating visualizations...")

# SST Distribution
df["sst"].hist(bins=40, color="steelblue", edgecolor="white")
plt.title("SST Distribution")
plt.xlabel("Sea Surface Temperature (°C)")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig("sst_distribution.png", dpi=150)
plt.clf()

# Correlation Heatmap
plt.figure(figsize=(12, 9))
num_df = df[[
    "latitude", "longitude", "ssta",
    "month", "year", "quarter", "day_of_year",
    "lat_grid", "lon_grid", "lat_x_month", "lon_x_month",
    "lat_squared", "ssta_abs", "sst"
]]
sns.heatmap(num_df.corr(), annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.savefig("correlation.png", dpi=150)
plt.clf()

# Monthly Average SST
monthly_avg = df.groupby("month")["sst"].mean()
monthly_avg.plot(marker="o", color="tomato", linewidth=2)
plt.title("Average SST by Month")
plt.xlabel("Month")
plt.ylabel("SST (°C)")
plt.xticks(range(1, 13))
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("sst_monthly.png", dpi=150)
plt.clf()

# SST vs Latitude
plt.figure(figsize=(6, 5))
plt.scatter(df["latitude"], df["sst"], alpha=0.2, s=5, color="teal")
plt.title("SST vs Latitude")
plt.xlabel("Latitude (°N)")
plt.ylabel("SST (°C)")
plt.tight_layout()
plt.savefig("sst_vs_latitude.png", dpi=150)
plt.clf()

# Yearly Average SST trend
yearly_avg = df.groupby("year")["sst"].mean()
yearly_avg.plot(marker="o", color="purple", linewidth=2)
plt.title("Yearly Average SST")
plt.xlabel("Year")
plt.ylabel("SST (°C)")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("sst_yearly.png", dpi=150)
plt.clf()

print("Visualizations saved.")

# =========================
# 4. FEATURES & TARGET
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

TARGET_COL = "sst"

X = df[FEATURE_COLS]
y = df[TARGET_COL]

# =========================
# 5. SCALING
# =========================

scaler_X = StandardScaler()
X_scaled = scaler_X.fit_transform(X)

# =========================
# 6. TRAIN / TEST SPLIT
# =========================

# shuffle=True: data is spatially ordered (same timestamp, many locations)
# shuffling ensures train/test see the same distribution of locations & times
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y.values,
    test_size=0.2,
    shuffle=True,
    random_state=42
)

print(f"\nTrain: {X_train.shape[0]:,} samples | Test: {X_test.shape[0]:,} samples")
print(f"Train SST mean: {y_train.mean():.2f} | Test SST mean: {y_test.mean():.2f}")

# =========================
# 7. DEFINE MODELS
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
# 8. TRAIN & EVALUATE
# =========================

results = {}

print("\n" + "=" * 55)
print("  MODEL TRAINING & EVALUATION")
print("=" * 55)

for name, model in models.items():
    print(f"\nTraining: {name} ...")
    model.fit(X_train, y_train)

    train_preds = model.predict(X_train)
    test_preds  = model.predict(X_test)

    train_mae  = mean_absolute_error(y_train, train_preds)
    test_mae   = mean_absolute_error(y_test,  test_preds)
    train_rmse = np.sqrt(mean_squared_error(y_train, train_preds))
    test_rmse  = np.sqrt(mean_squared_error(y_test,  test_preds))
    test_r2    = r2_score(y_test, test_preds)

    results[name] = {
        "model":       model,
        "test_preds":  test_preds,
        "train_mae":   train_mae,
        "test_mae":    test_mae,
        "train_rmse":  train_rmse,
        "test_rmse":   test_rmse,
        "test_r2":     test_r2
    }

    ratio = train_rmse / test_rmse
    print(f"  Train MAE  : {train_mae:.3f}  |  Test MAE  : {test_mae:.3f}")
    print(f"  Train RMSE : {train_rmse:.3f}  |  Test RMSE : {test_rmse:.3f}")
    print(f"  Test  R²   : {test_r2:.4f}")
    print(f"  Overfit ratio (train/test RMSE): {ratio:.2f}  {'✅ Good' if ratio >= 0.75 else '⚠️  Overfitting'}")

# =========================
# 9. MODEL COMPARISON PLOT
# =========================

model_names = list(results.keys())
colors = ["#2196F3", "#FF5722"]

fig, axes = plt.subplots(1, 3, figsize=(13, 4))
axes[0].bar(model_names, [results[m]["test_mae"]  for m in model_names], color=colors, edgecolor="white")
axes[0].set_title("Test MAE (lower = better)"); axes[0].set_ylabel("MAE")

axes[1].bar(model_names, [results[m]["test_rmse"] for m in model_names], color=colors, edgecolor="white")
axes[1].set_title("Test RMSE (lower = better)"); axes[1].set_ylabel("RMSE")

axes[2].bar(model_names, [results[m]["test_r2"]   for m in model_names], color=colors, edgecolor="white")
axes[2].set_title("Test R² (higher = better)"); axes[2].set_ylabel("R²"); axes[2].set_ylim(0, 1)

plt.suptitle("Model Comparison — SST Prediction", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("model_comparison.png", dpi=150)
plt.clf()
print("\nSaved: model_comparison.png")

# =========================
# 10. ACTUAL vs PREDICTED
# =========================

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for i, name in enumerate(model_names):
    preds = results[name]["test_preds"]
    axes[i].scatter(y_test[:2000], preds[:2000], alpha=0.4, s=10, color=colors[i])
    lims = [min(y_test.min(), preds.min()), max(y_test.max(), preds.max())]
    axes[i].plot(lims, lims, "k--", linewidth=1)
    axes[i].set_title(f"{name}\nR²={results[name]['test_r2']:.4f}")
    axes[i].set_xlabel("Actual SST (°C)")
    axes[i].set_ylabel("Predicted SST (°C)")

plt.suptitle("Actual vs Predicted SST", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("actual_vs_predicted.png", dpi=150)
plt.clf()
print("Saved: actual_vs_predicted.png")

# =========================
# 11. FEATURE IMPORTANCE
# =========================

for name in model_names:
    importances = results[name]["model"].feature_importances_
    indices = np.argsort(importances)[::-1]
    plt.figure(figsize=(13, 5))
    plt.bar(
        [FEATURE_COLS[i] for i in indices],
        importances[indices],
        color="#5C6BC0", edgecolor="white"
    )
    plt.title(f"Feature Importance — {name}")
    plt.ylabel("Importance")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    fname = f"feature_importance_{name.lower().replace(' ', '_')}.png"
    plt.savefig(fname, dpi=150)
    plt.clf()
    print(f"Saved: {fname}")

# =========================
# 12. RESIDUALS PLOT
# =========================

best_name = min(results, key=lambda m: results[m]["test_rmse"])
residuals = y_test - results[best_name]["test_preds"]
plt.figure(figsize=(8, 4))
plt.scatter(results[best_name]["test_preds"][:3000], residuals[:3000],
            alpha=0.3, s=8, color="purple")
plt.axhline(0, color="black", linewidth=1, linestyle="--")
plt.title(f"Residuals — {best_name}")
plt.xlabel("Predicted SST (°C)")
plt.ylabel("Residual (°C)")
plt.tight_layout()
plt.savefig("residuals.png", dpi=150)
plt.clf()
print("Saved: residuals.png")

# =========================
# 13. PICK BEST MODEL
# =========================

best_model = results[best_name]["model"]

print("\n" + "=" * 55)
print(f"  🏆 BEST MODEL  : {best_name}")
print(f"     Test RMSE  : {results[best_name]['test_rmse']:.3f}")
print(f"     Test MAE   : {results[best_name]['test_mae']:.3f}")
print(f"     Test R²    : {results[best_name]['test_r2']:.4f}")
print("=" * 55)

# =========================
# 14. SAVE ARTIFACTS
# =========================

pickle.dump(best_model, open("best_model.pkl",      "wb"))
pickle.dump(scaler_X,   open("scaler_X.pkl",        "wb"))

with open("best_model_name.txt", "w") as f:
    f.write(best_name)

for name, res in results.items():
    fname = name.lower().replace(" ", "_") + "_model.pkl"
    pickle.dump(res["model"], open(fname, "wb"))
    print(f"Saved: {fname}")

# Save feature column list for app.py
with open("feature_cols.txt", "w") as f:
    f.write("\n".join(FEATURE_COLS))

print("Saved: best_model.pkl")
print("Saved: scaler_X.pkl")
print("Saved: best_model_name.txt")
print("Saved: feature_cols.txt")
print("\n🎉 Training complete!")