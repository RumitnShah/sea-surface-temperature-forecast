"""Convert the monthly ERSST NetCDF files in ``files/`` into ``sst_grid.npz``.

Run this again whenever you add new monthly files:
    python prepare_data.py

Files are expected to be named like ``ersst.v6.202401.nc`` (NOAA's naming).
Download: https://www.ncei.noaa.gov/pub/data/cmb/ersst/v6/netcdf/
"""

import re
from pathlib import Path

import numpy as np
import xarray as xr

from sst_data import GRID_FILE, make_data

DATA_FOLDER = Path(__file__).parent / "files"


def main():
    files = sorted(DATA_FOLDER.glob("*.nc"))
    if not files:
        raise SystemExit(f"No .nc files found in {DATA_FOLDER}")

    grids, months, lat, lon = [], [], None, None
    for path in files:
        match = re.search(r"(\d{4})(\d{2})\.nc$", path.name)
        if not match:
            print(f"Skipping {path.name}: no YYYYMM in the file name")
            continue
        with xr.open_dataset(path, decode_times=False) as ds:
            sst = ds["sst"].squeeze().transpose("lat", "lon").values.astype("float32")
            lat, lon = ds["lat"].values, ds["lon"].values
        sst[sst < -5] = np.nan          # land / ice fill values
        grids.append(sst)
        months.append(f"{match.group(1)}-{match.group(2)}")

    np.savez_compressed(GRID_FILE, sst=np.stack(grids), time=np.array(months), lat=lat, lon=lon)

    data = make_data(np.stack(grids), months, lat, lon)
    print(f"Saved {GRID_FILE}: {len(months)} months, {months[0]} to {months[-1]}, "
          f"{data.sst.shape[1]} ocean cells")
    missing = data.missing_months()
    if missing:
        print(f"Missing months ({len(missing)}), filled by interpolation when training:")
        print("  " + ", ".join(missing))


if __name__ == "__main__":
    main()
