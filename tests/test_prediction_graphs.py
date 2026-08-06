import unittest
import sys
import types

streamlit = types.ModuleType("streamlit")
streamlit.cache_resource = lambda func: func
sys.modules.setdefault("streamlit", streamlit)
matplotlib = types.ModuleType("matplotlib")
pyplot = types.ModuleType("matplotlib.pyplot")
matplotlib.pyplot = pyplot
sys.modules.setdefault("matplotlib", matplotlib)
sys.modules.setdefault("matplotlib.pyplot", pyplot)
from main import monthly_prediction_series


class IdentityScaler:
    def transform(self, features):
        return features


class MonthEchoModel:
    def predict(self, features):
        return [float(features[0][12] / features[0][0])]


class MonthlyPredictionSeriesTest(unittest.TestCase):
    def test_returns_one_prediction_for_each_month_in_order(self):
        series = monthly_prediction_series(
            model=MonthEchoModel(),
            scaler=IdentityScaler(),
            lat=10.0,
            lon=70.0,
            year=2024,
        )

        self.assertEqual(list(series["Month"]), list(range(1, 13)))
        self.assertEqual(list(series["Predicted SST"]), [float(month) for month in range(1, 13)])


if __name__ == "__main__":
    unittest.main()
