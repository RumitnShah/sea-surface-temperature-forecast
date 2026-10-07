import unittest

import numpy as np

from features import FEATURE_NAMES, MAX_LEAD, N_LAGS, build_features
from forecast import Forecaster
from indicators import get_region
from sst_data import (anomalies, climatology, fill_gaps, make_data,
                      month_label, nearest_ocean_cell, to_lon360)


def synthetic_data(skip=("2021-06",)):
    """4 years on a tiny 3x4 grid: seasonal cycle + a per-cell offset; one land cell."""
    lat = np.array([-2.0, 0.0, 2.0])
    lon = np.array([0.0, 2.0, 4.0, 358.0])
    months, grids = [], []
    for year in range(2020, 2024):
        for month in range(1, 13):
            label = f"{year}-{month:02d}"
            if label in skip:
                continue
            grid = (20 + 3 * np.sin(2 * np.pi * month / 12)
                    + np.arange(12, dtype="float32").reshape(3, 4))
            grid[1, 1] = np.nan
            months.append(label)
            grids.append(grid)
    return make_data(np.stack(grids), months, lat, lon)


class LastAnomalyModel:
    """Predicts that the latest anomaly persists."""
    def predict(self, X):
        return X[:, 0]


class DataTest(unittest.TestCase):
    def setUp(self):
        self.data = synthetic_data()

    def test_gap_becomes_unobserved_row_and_land_is_dropped(self):
        self.assertEqual(self.data.missing_months(), ["2021-06"])
        self.assertEqual(len(self.data.month_ids), 48)
        self.assertEqual(self.data.sst.shape, (48, 11))

    def test_anomalies_of_a_pure_seasonal_cycle_are_zero(self):
        anom = anomalies(self.data, climatology(self.data))
        self.assertTrue(np.allclose(anom[self.data.observed], 0, atol=1e-4))
        self.assertTrue(np.isnan(anom[~self.data.observed]).all())

    def test_fill_gaps_interpolates_between_neighbours(self):
        values = np.array([[0.0], [np.nan], [np.nan], [3.0]])
        filled = fill_gaps(values, np.array([True, False, False, True]))
        self.assertTrue(np.allclose(filled[:, 0], [0, 1, 2, 3]))

    def test_longitude_conversion_and_nearest_cell(self):
        self.assertEqual(to_lon360(-2), 358.0)
        cell, dist = nearest_ocean_cell(self.data, 2.0, -2.0)
        self.assertEqual((self.data.cell_lat[cell], self.data.cell_lon[cell]), (2.0, 358.0))
        self.assertEqual(dist, 0.0)
        land_cell, land_dist = nearest_ocean_cell(self.data, 0.0, 2.0)   # the land cell
        self.assertGreater(land_dist, 0)


class FeatureTest(unittest.TestCase):
    def test_shape_order_and_lead(self):
        data = synthetic_data(skip=())
        anom = np.arange(48 * 11, dtype="float32").reshape(48, 11)
        X = build_features(anom, origin=20, lead=5, cells=[0, 3], data=data)
        self.assertEqual(X.shape, (2, len(FEATURE_NAMES)))
        self.assertEqual(X[0, 0], anom[20, 0])                 # lag 0 = latest month
        self.assertEqual(X[1, N_LAGS - 1], anom[20 - N_LAGS + 1, 3])
        self.assertEqual(X[0, FEATURE_NAMES.index("lead")], 5)

    def test_needs_enough_history(self):
        data = synthetic_data(skip=())
        with self.assertRaises(ValueError):
            build_features(np.zeros((48, 11)), origin=5, lead=1, cells=[0], data=data)


class ForecasterTest(unittest.TestCase):
    def setUp(self):
        self.data = synthetic_data()
        bundle = {"model": LastAnomalyModel(), "model_name": "dummy",
                  "climatology": climatology(self.data), "max_lead": MAX_LEAD,
                  "metrics": {"rmse_by_lead": {"1": 0.1}}}
        self.fc = Forecaster(self.data, bundle)

    def test_point_forecast_follows_the_seasonal_cycle(self):
        result = self.fc.forecast_point(2.0, 4.0)
        self.assertEqual(self.fc.last_month, "2023-12")
        self.assertEqual(result["months"][0], "2024-01")
        self.assertEqual(result["months"][-1], "2024-12")
        self.assertEqual(len(result["sst"]), MAX_LEAD)
        # zero anomaly -> forecast equals the usual value for that month
        self.assertTrue(np.allclose(result["sst"], result["normal"], atol=1e-3))
        self.assertEqual(result["typical_error"][0], 0.1)
        self.assertIsNone(result["typical_error"][1])

    def test_map_keeps_land_empty(self):
        sst, anomaly = self.fc.forecast_map(3)
        self.assertEqual(sst.shape, (3, 4))
        self.assertTrue(np.isnan(sst[1, 1]))
        self.assertEqual(int(np.isfinite(sst).sum()), 11)

    def test_rejects_leads_out_of_range(self):
        with self.assertRaises(ValueError):
            self.fc.predict_cells(MAX_LEAD + 1, np.array([0]))


class RegionTest(unittest.TestCase):
    def test_specific_seas_win_over_oceans(self):
        self.assertEqual(get_region(16, 66), "Arabian Sea")
        self.assertEqual(get_region(36, 18), "Mediterranean Sea")
        self.assertEqual(get_region(12, 114), "South China Sea")
        self.assertEqual(get_region(0, 200), "Pacific Ocean")     # 200E = 160W
        self.assertEqual(get_region(0, -30), "Atlantic Ocean")


if __name__ == "__main__":
    unittest.main()
