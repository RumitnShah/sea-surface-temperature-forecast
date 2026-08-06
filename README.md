# Marine Water Quality Classification

An end-to-end machine learning project for sea surface temperature (SST) prediction and marine water quality exploration.

This repository combines data conversion from NetCDF files, feature engineering, model training, artifact saving, and a Streamlit app for interactive prediction and regional analysis.

## What the project does

- Converts monthly ERSST NetCDF files into a tabular dataset.
- Engineers spatial and temporal features for SST modeling.
- Trains and compares Random Forest and XGBoost regressors.
- Saves the best model, scaler, and feature schema for reuse.
- Provides a Streamlit interface for SST prediction, monthly trends, and heatmap visualization.

## Tech Stack

- Python
- pandas, numpy
- scikit-learn
- XGBoost
- Streamlit
- matplotlib, seaborn
- xarray, netCDF4

## Project Structure

- `coversion.py` - converts NetCDF inputs into `sst_combined.csv`
- `new_train_model.py` - current training pipeline and artifact generation
- `train_model.py` - legacy training pipeline for the original CSV workflow
- `main.py` - Streamlit app for prediction and heatmap exploration
- `artifacts.py` - helper for reading the saved model name safely

## How to Run

1. Create and activate a virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Train the model and generate artifacts:

```bash
python new_train_model.py
```

4. Launch the Streamlit app:

```bash
streamlit run main.py
```

## Outputs

The training pipeline produces:

- `best_model.pkl`
- `scaler_X.pkl`
- `feature_cols.txt`
- evaluation plots such as `model_comparison.png`, `actual_vs_predicted.png`, and `residual_plot.png`