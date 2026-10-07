"""Loading the gridded monthly SST history and turning it into anomalies.

The history lives in ``sst_grid.npz`` (built by ``prepare_data.py``):
    sst   (n_months, n_lat, n_lon)  degrees C, NaN over land / ice
    time  (n_months,)               "YYYY-MM" strings (only months we have)
    lat   (n_lat,)                  -88 .. 88
    lon   (n_lon,)                  0 .. 358  (degrees east)
"""

from dataclasses import dataclass

import numpy as np

GRID_FILE = "sst_grid.npz"


# ---------- month helpers (a month is stored as year * 12 + month - 1) ----------

def month_id(year, month):
    return int(year) * 12 + int(month) - 1


def year_month(mid):
    year, month0 = divmod(int(mid), 12)
    return year, month0 + 1


def month_label(mid):
    year, month = year_month(mid)
    return f"{year}-{month:02d}"


def calendar_month(mid):
    """Calendar month 1..12 for a month id (works on arrays too)."""
    return np.asarray(mid) % 12 + 1


# ---------- data container ----------

@dataclass
class SSTData:
    sst: np.ndarray         # (n_months, n_ocean); rows of NaN for months with no file
    month_ids: np.ndarray   # (n_months,) continuous, no gaps
    observed: np.ndarray    # (n_months,) True where a real file exists
    lat: np.ndarray         # grid latitudes
    lon: np.ndarray         # grid longitudes, 0..360
    ocean_mask: np.ndarray  # (n_lat, n_lon) True for ocean cells
    cell_lat: np.ndarray    # (n_ocean,)
    cell_lon: np.ndarray    # (n_ocean,)

    @property
    def last_observed(self):
        return int(np.flatnonzero(self.observed)[-1])

    def missing_months(self):
        return [month_label(m) for m in self.month_ids[~self.observed]]

    def to_grid(self, values):
        """Put a (n_ocean,) vector back on the (n_lat, n_lon) map; land is NaN."""
        grid = np.full(self.ocean_mask.shape, np.nan, dtype="float32")
        grid[self.ocean_mask] = values
        return grid


def make_data(sst_grid, months, lat, lon):
    """Build SSTData from a (n_months, n_lat, n_lon) array and "YYYY-MM" labels."""
    ids = np.array([month_id(*str(m).split("-")) for m in months])
    order = np.argsort(ids)
    ids, sst_grid = ids[order], np.asarray(sst_grid, dtype="float32")[order]
    if len(np.unique(ids)) != len(ids):
        raise ValueError("Duplicate months in the SST grid.")

    ocean_mask = np.isfinite(sst_grid).all(axis=0)
    all_ids = np.arange(ids[0], ids[-1] + 1)
    observed = np.isin(all_ids, ids)

    sst = np.full((len(all_ids), int(ocean_mask.sum())), np.nan, dtype="float32")
    sst[observed] = sst_grid[:, ocean_mask]

    lat2d, lon2d = np.meshgrid(lat, lon, indexing="ij")
    return SSTData(
        sst=sst, month_ids=all_ids, observed=observed,
        lat=np.asarray(lat, dtype=float), lon=np.asarray(lon, dtype=float),
        ocean_mask=ocean_mask,
        cell_lat=lat2d[ocean_mask].astype("float32"),
        cell_lon=lon2d[ocean_mask].astype("float32"),
    )


def load_grid(path=GRID_FILE):
    with np.load(path) as f:
        return make_data(f["sst"], f["time"], f["lat"], f["lon"])


# ---------- climatology and anomalies ----------

def climatology(data, last_month_id=None):
    """Average SST for each calendar month and cell -> (12, n_ocean).

    Pass ``last_month_id`` to use only months up to that point (so a test
    period never leaks into the climatology used for training).
    """
    use = data.observed.copy()
    if last_month_id is not None:
        use &= data.month_ids <= last_month_id
    cal = calendar_month(data.month_ids)
    clim = np.empty((12, data.sst.shape[1]), dtype="float32")
    for m in range(1, 13):
        rows = use & (cal == m)
        if not rows.any():
            raise ValueError(f"No data for calendar month {m}; cannot build a climatology.")
        clim[m - 1] = data.sst[rows].mean(axis=0)
    return clim


def anomalies(data, clim):
    """SST minus its usual value for that month and place. Missing months stay NaN."""
    return data.sst - clim[calendar_month(data.month_ids) - 1]


def fill_gaps(values, observed):
    """Fill missing months by linear interpolation in time between real months."""
    filled = values.copy()
    obs_idx = np.flatnonzero(observed)
    for t in np.flatnonzero(~observed):
        before, after = obs_idx[obs_idx < t], obs_idx[obs_idx > t]
        if len(before) and len(after):
            a, b = before[-1], after[0]
            w = (t - a) / (b - a)
            filled[t] = (1 - w) * values[a] + w * values[b]
        else:
            filled[t] = values[before[-1] if len(before) else after[0]]
    return filled


# ---------- locations ----------

def to_lon360(lon):
    """Convert any longitude (-180..180 or 0..360) to the grid's 0..360 range."""
    return float(lon) % 360.0


def nearest_ocean_cell(data, lat, lon):
    """Index of the closest ocean cell and its distance in degrees."""
    dlon = np.abs(data.cell_lon - to_lon360(lon))
    dlon = np.minimum(dlon, 360.0 - dlon)
    dist = np.hypot(data.cell_lat - float(lat), dlon)
    cell = int(np.argmin(dist))
    return cell, float(dist[cell])
