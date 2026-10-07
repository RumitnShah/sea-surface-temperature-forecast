"""Feature building shared by training and the app (one definition, no drift)."""

import numpy as np

from sst_data import calendar_month

N_LAGS = 12    # months of history the model looks at
MAX_LEAD = 12  # how many months ahead it can forecast

FEATURE_NAMES = (
    [f"anom_lag_{k}" for k in range(N_LAGS)]          # lag 0 = latest known month
    + ["anom_mean_3", "anom_mean_12",
       "lat", "lon_sin", "lon_cos",
       "target_month_sin", "target_month_cos",
       "lead"]
)


def build_features(anom, origin, lead, cells, data):
    """Features for forecasting month ``origin + lead`` from data up to ``origin``.

    anom   : (n_months, n_ocean) gap-filled anomalies
    origin : row index of the latest known month
    lead   : months ahead (1..MAX_LEAD)
    cells  : indices of the ocean cells to build rows for
    """
    if origin < N_LAGS - 1:
        raise ValueError(f"Need {N_LAGS} months of history before the forecast start.")
    cells = np.asarray(cells)
    history = anom[origin - N_LAGS + 1: origin + 1][:, cells]   # oldest -> newest
    lags = history[::-1].T                                       # (n_cells, N_LAGS), newest first

    target_month = calendar_month(data.month_ids[origin] + lead)
    angle = 2 * np.pi * target_month / 12
    lon_rad = np.deg2rad(data.cell_lon[cells])
    n = len(cells)

    return np.column_stack([
        lags,
        lags[:, :3].mean(axis=1),
        lags.mean(axis=1),
        data.cell_lat[cells],
        np.sin(lon_rad),
        np.cos(lon_rad),
        np.full(n, np.sin(angle)),
        np.full(n, np.cos(angle)),
        np.full(n, lead),
    ]).astype("float32")
