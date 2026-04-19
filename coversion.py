import xarray as xr
import pandas as pd
import numpy as np
import os

# 📂 Folder where your .nc files are stored
data_folder = r"C:\Users\HP\OneDrive\Documents\GitHub\marine-water-quality-classification\files"   # change this path

# 📁 Output CSV file
output_csv = "sst_combined.csv"

# 🔍 Get all .nc files
files = [os.path.join(data_folder, f) for f in os.listdir(data_folder) if f.endswith(".nc")]

print(f"Found {len(files)} files...")

# 📊 Load multiple NetCDF files
ds = xr.open_mfdataset(files, combine='by_coords')

print("Dataset loaded!")

# 🔄 Convert to DataFrame
df = ds.to_dataframe().reset_index()

print("Converted to DataFrame!")

# ✅ Keep only required columns
df = df[["time", "lat", "lon", "sst"]]

# ❌ Remove missing values
df = df.dropna()

# 🔧 Fix longitude (0–360 → -180 to 180)
df["lon"] = df["lon"].apply(lambda x: x - 360 if x > 180 else x)

# 🧠 Convert time
df["time"] = pd.to_datetime(df["time"])

# 📅 Feature Engineering
df["year"] = df["time"].dt.year
df["month"] = df["time"].dt.month
df["dayofyear"] = df["time"].dt.dayofyear

# 🔁 Cyclical features
df["sin_month"] = np.sin(2 * np.pi * df["month"] / 12)
df["cos_month"] = np.cos(2 * np.pi * df["month"] / 12)

# 🌍 Latitude importance
df["abs_lat"] = np.abs(df["lat"])

# ⚡ Optional: reduce dataset size (IMPORTANT if large)
df = df.sample(min(100000, len(df)), random_state=42)  # adjust size if needed

# 💾 Save CSV
df.to_csv(output_csv, index=False)

print(f"✅ CSV saved as: {output_csv}")
print(df.head())