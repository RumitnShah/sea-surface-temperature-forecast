# Sea Surface Temperature Forecasting

Forecasts future monthly sea surface temperature (SST) from past values, anywhere in the ocean, up to 12 months ahead. Includes a Streamlit app with a point forecast, a global forecast map and an accuracy page.

Data: NOAA ERSST v6 monthly SST on a 2° grid (2015 onwards).

## How it works

1. **Anomalies.** For each ocean grid cell, SST is turned into an anomaly: the difference from that cell's usual temperature for that calendar month.
2. **Learning from the past.** The model sees a cell's last 12 monthly anomalies (plus location, target month and how far ahead it is forecasting) and predicts the anomaly 1 to 12 months later.
3. **Forecast.** Forecast SST = usual temperature for the target month + predicted anomaly.

The model is tested on the most recent period (2024 onwards), which it never sees in training, and compared against two simple baselines:

- **Climatology**: assume the usual temperature for that month.
- **Persistence**: assume the current anomaly stays the same.

## Results

Reference run (scikit-learn gradient boosting standing in for XGBoost), test period 2024-01 to 2026-03:

| Model | RMSE (°C) | MAE (°C) |
|---|---|---|
| Gradient boosting | 0.485 | 0.335 |
| Random Forest | 0.491 | 0.336 |
| Ridge | 0.502 | 0.346 |
| Climatology (baseline) | 0.531 | 0.367 |
| Persistence (baseline) | 0.602 | 0.412 |

Error by lead time for the best model: 0.36 °C at 1 month, 0.45 °C at 3 months, 0.49 °C at 6 months, 0.53 °C at 12 months. The model is clearly better than the baselines for the next few months; by 12 months ahead it is about as good as climatology. Run `python train_model.py` to produce your own numbers in `metrics.json` and `plots/`.

## Project structure

- `prepare_data.py` - converts the NetCDF files in `files/` into `sst_grid.npz`
- `sst_data.py` - loads the grid, computes climatology and anomalies, fills gaps
- `features.py` - feature building, shared by training and the app
- `train_model.py` - trains and compares models, saves the best one
- `forecast.py` - loads the model and produces forecasts
- `indicators.py` - rule-of-thumb temperature labels (region, coral heat stress, species ranges)
- `main.py` - Streamlit app
- `tests/` - unit tests

## How to run

```bash
pip install -r requirements.txt
python train_model.py        # a few minutes; writes sst_forecaster.pkl, metrics.json, plots/
streamlit run main.py
```

Run the tests with `python -m unittest discover -s tests -t .`

## Adding more data

Download monthly files named `ersst.v6.YYYYMM.nc` from
<https://www.ncei.noaa.gov/pub/data/cmb/ersst/v6/netcdf/> into `files/`, then:

```bash
python prepare_data.py
python train_model.py
```

Forecasts always start from the latest month in `files/`.

## Known limitations

- **Missing months.** 19 months have no file and are filled by interpolation: 2018-02, 2018-12, 2020-10, 2021-07, 2021-10, 2021-11, all of 2022, and 2025-07. Adding them should improve accuracy.
- **Short history.** About ten years of data is little for learning slow patterns such as El Niño. Adding earlier years (ERSST goes back to 1854) would help most.
- **Indicators are not water quality measurements.** The coral, species and comfort labels in the app are fixed temperature thresholds applied to the forecast.
