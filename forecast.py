"""Use the trained model to forecast SST for the months after the last observation."""

import joblib
import numpy as np

from features import MAX_LEAD, build_features
from sst_data import (GRID_FILE, anomalies, calendar_month, fill_gaps,
                      load_grid, month_label, nearest_ocean_cell)

MODEL_FILE = "sst_forecaster.pkl"


class Forecaster:
    def __init__(self, data, bundle):
        self.data = data
        self.model = bundle["model"]
        self.model_name = bundle["model_name"]
        self.metrics = bundle.get("metrics", {})
        self.clim = bundle["climatology"]
        if self.clim.shape[1] != data.sst.shape[1]:
            raise ValueError("Model and SST grid do not match. Re-run train_model.py.")
        self.anom = fill_gaps(anomalies(data, self.clim), data.observed)
        self.origin = data.last_observed
        self.max_lead = bundle.get("max_lead", MAX_LEAD)

    @classmethod
    def load(cls, grid_path=GRID_FILE, model_path=MODEL_FILE):
        return cls(load_grid(grid_path), joblib.load(model_path))

    @property
    def last_month(self):
        return month_label(self.data.month_ids[self.origin])

    def target_label(self, lead):
        return month_label(self.data.month_ids[self.origin] + lead)

    def _climatology_for(self, lead, cells):
        month = calendar_month(self.data.month_ids[self.origin] + lead)
        return self.clim[month - 1][cells]

    def predict_cells(self, lead, cells):
        """Forecast SST (deg C) ``lead`` months after the last observation."""
        if not 1 <= lead <= self.max_lead:
            raise ValueError(f"lead must be between 1 and {self.max_lead}")
        X = build_features(self.anom, self.origin, lead, cells, self.data)
        return self._climatology_for(lead, cells) + self.model.predict(X)

    def forecast_point(self, lat, lon):
        """Forecast every lead for the ocean cell nearest to (lat, lon).

        Returns a dict with the cell used, recent history and the forecast.
        """
        cell, distance = nearest_ocean_cell(self.data, lat, lon)
        cells = np.array([cell])
        leads = list(range(1, self.max_lead + 1))
        rmse_by_lead = self.metrics.get("rmse_by_lead", {})
        return {
            "cell_lat": float(self.data.cell_lat[cell]),
            "cell_lon": float(self.data.cell_lon[cell]),
            "distance_deg": distance,
            "history_months": [month_label(m) for m in self.data.month_ids],
            "history_sst": self.data.sst[:, cell],          # NaN where a month is missing
            "leads": leads,
            "months": [self.target_label(h) for h in leads],
            "sst": [float(self.predict_cells(h, cells)[0]) for h in leads],
            "normal": [float(self._climatology_for(h, cells)[0]) for h in leads],
            "typical_error": [rmse_by_lead.get(str(h)) for h in leads],
        }

    def forecast_map(self, lead):
        """Forecast for the whole ocean -> (sst_grid, anomaly_grid), NaN over land."""
        cells = np.arange(self.data.sst.shape[1])
        sst = self.predict_cells(lead, cells)
        return self.data.to_grid(sst), self.data.to_grid(sst - self._climatology_for(lead, cells))
