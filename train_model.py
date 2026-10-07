"""Train a model that forecasts future sea surface temperature from past values.

    python train_model.py

How it works
    1. For every ocean cell we turn SST into an anomaly: the difference from
       that cell's usual temperature for that calendar month.
    2. The model sees the last 12 monthly anomalies of a cell and predicts the
       anomaly 1 to 12 months later. Forecast SST = usual temperature + anomaly.
    3. The test set is the most recent period, which the model never saw. Each
       model is compared against two simple baselines:
         - "Climatology": assume the usual temperature for that month
         - "Persistence": assume the current anomaly stays the same
"""

import json
import warnings
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from features import FEATURE_NAMES, MAX_LEAD, N_LAGS, build_features
from forecast import MODEL_FILE
from sst_data import (anomalies, climatology, fill_gaps, load_grid, month_id,
                      month_label)

try:
    from xgboost import XGBRegressor
except ImportError:  # the script still works without xgboost
    XGBRegressor = None

TEST_START = month_id(2024, 1)   # forecasts for this month and later are the test set
TRAIN_ROWS = 1_200_000           # rows sampled for training
TEST_ROWS = 600_000
PLOT_DIR = Path("plots")
METRICS_FILE = "metrics.json"
SEED = 42


def candidate_models():
    models = {
        "Ridge": make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
        "Random Forest": RandomForestRegressor(
            n_estimators=100, max_depth=14, min_samples_leaf=20,
            max_features=0.5, max_samples=0.15, n_jobs=-1, random_state=SEED),
    }
    if XGBRegressor is not None:
        models["XGBoost"] = XGBRegressor(
            n_estimators=400, learning_rate=0.05, max_depth=7,
            subsample=0.8, colsample_bytree=0.8, min_child_weight=20,
            tree_method="hist", n_jobs=-1, random_state=SEED, verbosity=0)
    else:
        warnings.warn("xgboost is not installed; using scikit-learn gradient boosting instead.")
        models["Gradient Boosting"] = HistGradientBoostingRegressor(
            max_iter=400, learning_rate=0.05, max_leaf_nodes=63,
            min_samples_leaf=50, random_state=SEED)
    return models


def forecast_blocks(data, first_target=None, last_target=None):
    """All (origin, lead) pairs whose start month and target month are real data."""
    blocks = []
    n_months = len(data.month_ids)
    for origin in range(N_LAGS - 1, n_months):
        if not data.observed[origin]:
            continue
        for lead in range(1, MAX_LEAD + 1):
            target = origin + lead
            if target >= n_months or not data.observed[target]:
                continue
            target_id = data.month_ids[target]
            if first_target is not None and target_id < first_target:
                continue
            if last_target is not None and target_id > last_target:
                continue
            blocks.append((origin, lead))
    return blocks


def make_samples(data, anom, anom_filled, blocks, total_rows, rng):
    """Sample cells from each block -> X, y (target anomaly), lead."""
    n_ocean = data.sst.shape[1]
    per_block = min(n_ocean, max(1, total_rows // len(blocks)))
    X, y, lead_col = [], [], []
    for origin, lead in blocks:
        cells = rng.choice(n_ocean, per_block, replace=False)
        X.append(build_features(anom_filled, origin, lead, cells, data))
        y.append(anom[origin + lead, cells])
        lead_col.append(np.full(per_block, lead))
    return np.vstack(X), np.concatenate(y), np.concatenate(lead_col)


def rmse(error):
    return float(np.sqrt(np.mean(np.square(error))))


def score(pred, y, leads):
    err = pred - y
    return {
        "rmse": rmse(err),
        "mae": float(np.mean(np.abs(err))),
        "rmse_by_lead": {str(h): rmse(err[leads == h]) for h in range(1, MAX_LEAD + 1)
                         if (leads == h).any()},
    }


def evaluate(data, rng):
    """Train on the past, test on the most recent period."""
    clim = climatology(data, last_month_id=TEST_START - 1)   # no test data in here
    anom = anomalies(data, clim)
    anom_filled = fill_gaps(anom, data.observed)

    train_blocks = forecast_blocks(data, last_target=TEST_START - 1)
    test_blocks = forecast_blocks(data, first_target=TEST_START)
    if not train_blocks or not test_blocks:
        raise SystemExit("Not enough data on one side of TEST_START to train and test.")

    X_train, y_train, _ = make_samples(data, anom, anom_filled, train_blocks, TRAIN_ROWS, rng)
    X_test, y_test, lead_test = make_samples(data, anom, anom_filled, test_blocks, TEST_ROWS, rng)
    print(f"Training rows: {len(y_train):,}   Test rows: {len(y_test):,}")
    print(f"Train: forecasts up to {month_label(TEST_START - 1)}   "
          f"Test: forecasts from {month_label(TEST_START)}")

    results = {
        "Climatology (baseline)": score(np.zeros_like(y_test), y_test, lead_test),
        "Persistence (baseline)": score(X_test[:, 0], y_test, lead_test),
    }
    predictions = {}
    for name, model in candidate_models().items():
        print(f"Training {name} ...")
        model.fit(X_train, y_train)
        predictions[name] = model.predict(X_test)
        results[name] = score(predictions[name], y_test, lead_test)

    print(f"\n{'Model':<26}{'RMSE (°C)':>10}{'MAE (°C)':>10}")
    for name, r in sorted(results.items(), key=lambda kv: kv[1]["rmse"]):
        print(f"{name:<26}{r['rmse']:>10.3f}{r['mae']:>10.3f}")

    best = min(predictions, key=lambda n: results[n]["rmse"])
    print(f"\nBest model: {best}")
    save_plots(results, best, predictions[best], y_test)
    return results, best


def save_plots(results, best, best_pred, y_test):
    PLOT_DIR.mkdir(exist_ok=True)
    names = sorted(results, key=lambda n: results[n]["rmse"])

    fig, ax = plt.subplots(figsize=(8, 4))
    colors = ["#9AA5B1" if "baseline" in n else "#0B7285" for n in names]
    ax.barh(names[::-1], [results[n]["rmse"] for n in names[::-1]], color=colors[::-1])
    ax.set_xlabel("Test RMSE (°C), lower is better")
    ax.set_title("Model comparison on the unseen test period")
    fig.tight_layout(); fig.savefig(PLOT_DIR / "model_comparison.png", dpi=120); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    for name in names:
        by_lead = results[name]["rmse_by_lead"]
        ax.plot([int(h) for h in by_lead], list(by_lead.values()), marker="o",
                linestyle="--" if "baseline" in name else "-",
                linewidth=2.5 if name == best else 1.5, label=name)
    ax.set_xlabel("Months ahead"); ax.set_ylabel("Test RMSE (°C)")
    ax.set_title("Forecast error grows with lead time")
    ax.set_xticks(range(1, MAX_LEAD + 1)); ax.grid(alpha=0.3); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(PLOT_DIR / "rmse_by_lead.png", dpi=120); plt.close(fig)

    show = np.random.default_rng(SEED).choice(len(y_test), min(20_000, len(y_test)), replace=False)
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.scatter(y_test[show], best_pred[show], s=3, alpha=0.2, color="#0B7285")
    lim = [float(y_test.min()), float(y_test.max())]
    ax.plot(lim, lim, color="black", linewidth=1)
    ax.set_xlabel("Actual anomaly (°C)"); ax.set_ylabel("Predicted anomaly (°C)")
    ax.set_title(f"{best}: actual vs predicted (test period)")
    fig.tight_layout(); fig.savefig(PLOT_DIR / "actual_vs_predicted.png", dpi=120); plt.close(fig)


def fit_final(data, best, rng):
    """Refit the winning model on all the data so forecasts use the latest months."""
    clim = climatology(data)
    anom = anomalies(data, clim)
    anom_filled = fill_gaps(anom, data.observed)
    X, y, _ = make_samples(data, anom, anom_filled, forecast_blocks(data), TRAIN_ROWS, rng)
    model = candidate_models()[best]
    print(f"Refitting {best} on all {len(y):,} rows ...")
    model.fit(X, y)
    return model, clim


def main():
    rng = np.random.default_rng(SEED)
    data = load_grid()
    print(f"Loaded {int(data.observed.sum())} months "
          f"({month_label(data.month_ids[0])} to {month_label(data.month_ids[-1])}), "
          f"{data.sst.shape[1]} ocean cells")
    missing = data.missing_months()
    if missing:
        print(f"Note: {len(missing)} months have no file and are interpolated: {', '.join(missing)}")

    results, best = evaluate(data, rng)
    model, clim = fit_final(data, best, rng)

    metrics = {
        "best_model": best,
        "test_period_start": month_label(TEST_START),
        "trained_through": month_label(data.month_ids[data.last_observed]),
        "missing_months": missing,
        "rmse_by_lead": results[best]["rmse_by_lead"],
        "results": results,
    }
    joblib.dump({
        "model": model, "model_name": best, "climatology": clim,
        "feature_names": FEATURE_NAMES, "max_lead": MAX_LEAD, "metrics": metrics,
    }, MODEL_FILE, compress=3)
    Path(METRICS_FILE).write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"\nSaved {MODEL_FILE}, {METRICS_FILE} and plots in {PLOT_DIR}/")


if __name__ == "__main__":
    main()
