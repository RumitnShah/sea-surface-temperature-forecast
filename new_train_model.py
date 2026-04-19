# =========================
# IMPORT LIBRARIES
# =========================

import pandas as pd              # For data handling
import numpy as np               # For numerical operations
import pickle                    # For saving model
import matplotlib.pyplot as plt  # For plotting
import seaborn as sns            # For visualization
import warnings
warnings.filterwarnings("ignore")

# Machine Learning libraries
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# Models
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor


# =========================
# 1. LOAD DATASET
# =========================

# Load NOAA SST dataset (converted from NetCDF to CSV)
df = pd.read_csv("sst_combined.csv")

# Rename columns to match existing model pipeline
df.rename(columns={
    "lat": "latitude",
    "lon": "longitude"
}, inplace=True)

# Add dummy anomaly column (not present in dataset but required by model)
df["ssta"] = 0


# =========================
# 2. DATA PREPROCESSING
# =========================

# Convert time column to datetime format
df["time"] = pd.to_datetime(df["time"], errors="coerce")

# Remove rows with missing values in time or target variable (sst)
df = df.dropna(subset=["time", "sst"])

# Extract time-based features
df["quarter"]     = df["time"].dt.quarter      # Season indicator
df["day_of_year"] = df["time"].dt.dayofyear    # Day position in year

# Use existing cyclical encoding for month
df["month_sin"] = df["sin_month"]
df["month_cos"] = df["cos_month"]

# Create cyclical encoding for day of year
df["doy_sin"] = np.sin(2 * np.pi * df["day_of_year"] / 365)
df["doy_cos"] = np.cos(2 * np.pi * df["day_of_year"] / 365)


# =========================
# 3. FEATURE ENGINEERING
# =========================

# Spatial grouping (helps model learn regional patterns)
df["lat_grid"] = (df["latitude"] / 2.0).round() * 2.0
df["lon_grid"] = (df["longitude"] / 2.0).round() * 2.0

# Interaction features (capture location + seasonal effects)
df["lat_x_month"] = df["latitude"] * df["month"]
df["lon_x_month"] = df["longitude"] * df["month"]

# Non-linear transformation of latitude
df["lat_squared"] = df["latitude"] ** 2

# Absolute anomaly (though anomaly is 0 here)
df["ssta_abs"] = df["ssta"].abs()

# Remove unrealistic SST values
df = df[(df["sst"] > -2) & (df["sst"] < 40)]

print(f"Dataset shape: {df.shape}")


# =========================
# 4. FEATURE SELECTION
# =========================

# List of input features
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

# Input features (X) and target variable (y)
X = df[FEATURE_COLS]
y = df["sst"]


# =========================
# 5. DATA SCALING
# =========================

# Standardize features (mean=0, std=1)
scaler_X = StandardScaler()
X_scaled = scaler_X.fit_transform(X)


# =========================
# 6. TRAIN-TEST SPLIT
# =========================

# Split dataset into training (80%) and testing (20%)
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

    # Random Forest: Ensemble of decision trees
    "Random Forest": RandomForestRegressor(
        n_estimators=300,       # Number of trees
        max_depth=20,           # Depth of trees
        min_samples_split=10,   # Minimum samples to split
        min_samples_leaf=5,     # Minimum samples at leaf
        max_features=0.6,       # Features used per split
        n_jobs=-1,              # Use all CPU cores
        random_state=42
    ),

    # XGBoost: Gradient boosting algorithm
    "XGBoost": XGBRegressor(
        n_estimators=600,
        learning_rate=0.03,
        max_depth=8,
        subsample=0.7,
        colsample_bytree=0.7,
        colsample_bylevel=0.7,
        reg_alpha=0.3,          # L1 regularization
        reg_lambda=2.0,         # L2 regularization
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

    # Train model
    model.fit(X_train, y_train)

    # Predictions
    train_preds = model.predict(X_train)
    test_preds  = model.predict(X_test)

    # Evaluation metrics
    train_rmse = np.sqrt(mean_squared_error(y_train, train_preds))
    test_rmse  = np.sqrt(mean_squared_error(y_test, test_preds))
    test_r2    = r2_score(y_test, test_preds)

    print(f"Train RMSE : {train_rmse:.3f}")
    print(f"Test RMSE  : {test_rmse:.3f}")
    print(f"R² Score   : {test_r2:.4f}")

    # ==========================
    # OVERFITTING / UNDERFITTING CHECK
    # ==========================

    if test_rmse > train_rmse * 1.5:
        print("⚠️ Overfitting detected")
    elif train_rmse > 3:
        print("⚠️ Underfitting detected")
    else:
        print("✅ Good model fit")

    # Store results
    results[name] = {
        "model": model,
        "test_rmse": test_rmse
    }


# =========================
# 9. SELECT BEST MODEL
# =========================

# Choose model with lowest test RMSE
best_name = min(results, key=lambda x: results[x]["test_rmse"])
best_model = results[best_name]["model"]

print(f"\n🏆 Best Model: {best_name}")


# =========================
# 10. SAVE MODEL
# =========================

# Save trained model and scaler for future use
pickle.dump(best_model, open("best_model.pkl", "wb"))
pickle.dump(scaler_X, open("scaler_X.pkl", "wb"))

# Save feature list
with open("feature_cols.txt", "w") as f:
    f.write("\n".join(FEATURE_COLS))

print("\n🎉 Training complete!")