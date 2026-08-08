"""
Training script for the 48h-ahead weather / flood models.

Design (differs from the previous version in three important ways)
------------------------------------------------------------------
1. **No calendar features reach any model.** `hour_*`, `doy_*` and `month` are
   excluded from the feature matrix, so a model physically cannot learn "August
   means flood". The season is reintroduced separately as a climatology table and
   blended at a fixed 50% weight (see `weather_features.blend_percent`), which is
   what keeps the sensors responsible for the other 50%.

2. **The flood and rainfall targets are continuous 0-100 indices**, trained with
   regressors, instead of three/four hard classes. Categories are derived from the
   final blended percentage at inference time, so the label and the number can
   never disagree.

3. **Chronologically blocked split with an embargo**, and the reported metrics are
   for the *blended* prediction — the number the API actually returns — not just
   the sensor model in isolation.

Usage:
    python train_models.py [--data weather_flood_dataset.csv.gz] [--out model_artifacts]
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn import __version__ as sklearn_version
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import RobustScaler

import weather_features as wf

try:
    import lightgbm as lgb
    from lightgbm import LGBMRegressor

    GBM_BACKEND = "lightgbm"
except ImportError:  # pragma: no cover - fallback for environments without lightgbm
    from sklearn.ensemble import GradientBoostingRegressor as LGBMRegressor

    lgb = None
    GBM_BACKEND = "sklearn_gbm"

warnings.filterwarnings("ignore")

RANDOM_STATE = 42
MODEL_VERSION = "v7_blended_index"

ALL_TARGETS = wf.REGRESSION_TARGETS + wf.INDEX_TARGETS


def load_dataset(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    for _, target_col, delta_col in wf.PRESSURE_PAIRS:
        now_col = target_col.replace("_target", "")
        df[delta_col] = df[target_col] - df[now_col]

    missing = [c for c in wf.SENSOR_FEATURE_COLUMNS + ALL_TARGETS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Dataset is missing required columns: {missing}. "
            "Regenerate it with `python generate_dataset.py`."
        )

    leaked = [c for c in wf.TIME_COLUMNS if c in wf.SENSOR_FEATURE_COLUMNS]
    if leaked:
        raise AssertionError(f"Calendar features leaked into the model inputs: {leaked}")

    return df


def chronological_split(df: pd.DataFrame, interval_min: float) -> pd.DataFrame:
    """
    Week-long blocks assigned round-robin to train/val/test, with a 48h embargo
    purged around every boundary so a row's 48h-ahead target cannot appear as a
    feature-time neighbour on the other side of the split.
    """
    steps_per_hour = 60.0 / interval_min
    block_steps = int(7 * 24 * steps_per_hour)
    embargo_steps = int(48 * steps_per_hour)

    block_id = np.arange(len(df)) // block_steps
    # Deterministic 70/15/15 rotation over whole weeks: keeps every season present
    # in all three splits without shuffling individual rows.
    pattern = ["train"] * 14 + ["val"] * 3 + ["test"] * 3
    split = np.array([pattern[b % len(pattern)] for b in block_id])

    boundaries = np.where(split[1:] != split[:-1])[0] + 1
    embargo = np.zeros(len(df), dtype=bool)
    for cp in boundaries:
        embargo[max(0, cp - embargo_steps) : min(len(df), cp + embargo_steps)] = True

    df = df.assign(split=split, embargo=embargo)
    print(
        f"  block_steps={block_steps} embargo_steps={embargo_steps} "
        f"purged={int(embargo.sum()):,} rows ({embargo.mean():.1%})"
    )
    return df.loc[~df["embargo"]].copy()


def fit_regressor(X_train, y_train, X_val, y_val, **overrides):
    params = dict(
        n_estimators=900,
        learning_rate=0.04,
        max_depth=6,
        subsample=0.85,
        random_state=RANDOM_STATE,
    )
    params.update(overrides)

    if GBM_BACKEND == "lightgbm":
        model = LGBMRegressor(**params, colsample_bytree=0.85, verbosity=-1)
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            eval_metric="l1",
            callbacks=[lgb.early_stopping(stopping_rounds=60, verbose=False)],
        )
    else:  # pragma: no cover
        model = LGBMRegressor(
            **params, validation_fraction=0.15, n_iter_no_change=40, tol=1e-4
        )
        model.fit(X_train, y_train)
    return model


def metrics_for(y_true, y_pred) -> dict:
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": float(r2_score(y_true, y_pred)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="weather_flood_dataset.csv.gz")
    parser.add_argument("--out", default="model_artifacts")
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Delete the output directory before writing new artifacts.",
    )
    args = parser.parse_args()

    print(f"Gradient boosting backend: {GBM_BACKEND}")
    df = load_dataset(args.data)
    interval_min = (
        df["timestamp"].diff().median().total_seconds() / 60.0
    )
    print(f"Dataset: {df.shape[0]:,} rows, sampling interval {interval_min:.0f} min")

    if args.clean and os.path.isdir(args.out):
        shutil.rmtree(args.out)
    os.makedirs(args.out, exist_ok=True)

    # --- Climatology: the calendar half of the prediction ----------------------
    # Built from the ground-truth index targets, i.e. the average seasonal risk
    # for each day of the year across all simulated years. This is an explicit
    # lookup table, not a model, so its influence is exactly CLIMATE_WEIGHT.
    print("\n--- Building climatology table ---")
    climatology = wf.build_climatology(
        df["timestamp"],
        {
            "flood_index": df["flood_index_target"],
            "rain_index": df["rain_index_target"],
            "temperature": df["temperature_target"],
        },
    )
    with open(os.path.join(args.out, "climatology.json"), "w") as fh:
        json.dump(climatology, fh)
    print(
        f"  flood_index climatology range: "
        f"{min(climatology['flood_index']):.1f} - {max(climatology['flood_index']):.1f}%"
    )

    # --- Split -----------------------------------------------------------------
    print("\n--- Splitting ---")
    df_split = chronological_split(df, interval_min)
    train_df = df_split[df_split["split"] == "train"]
    val_df = df_split[df_split["split"] == "val"]
    test_df = df_split[df_split["split"] == "test"]
    print(f"  train={len(train_df):,} val={len(val_df):,} test={len(test_df):,}")

    X_train_raw = train_df[wf.SENSOR_FEATURE_COLUMNS]
    scaler = RobustScaler().fit(X_train_raw)

    def scale(frame: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(
            scaler.transform(frame[wf.SENSOR_FEATURE_COLUMNS]),
            columns=wf.SENSOR_FEATURE_COLUMNS,
            index=frame.index,
        )

    X_train, X_val, X_test = scale(train_df), scale(val_df), scale(test_df)
    joblib.dump(scaler, os.path.join(args.out, "feature_scaler.joblib"))

    # Recorded so the API can clip out-of-distribution input instead of
    # extrapolating off the end of a tree.
    feature_ranges = {
        col: {
            "min": float(X_train_raw[col].min()),
            "max": float(X_train_raw[col].max()),
        }
        for col in wf.SENSOR_FEATURE_COLUMNS
    }

    # --- Train -----------------------------------------------------------------
    models, model_metrics = {}, {}
    print("\n===== Training sensor models (no calendar features) =====")
    for target in ALL_TARGETS:
        model = fit_regressor(X_train, train_df[target], X_val, val_df[target])
        preds = model.predict(X_test)
        if target in wf.INDEX_TARGETS:
            preds = np.clip(preds, 0.0, 100.0)
        models[target] = model
        model_metrics[target] = metrics_for(test_df[target], preds)
        joblib.dump(model, os.path.join(args.out, f"model_{target}.joblib"))
        m = model_metrics[target]
        print(f"  {target:28s} MAE={m['MAE']:.3f}  RMSE={m['RMSE']:.3f}  R2={m['R2']:.3f}")

        top = sorted(
            zip(wf.SENSOR_FEATURE_COLUMNS, model.feature_importances_),
            key=lambda pair: -pair[1],
        )[:5]
        print("      top features: " + ", ".join(f"{n}={v}" for n, v in top))

    print("\n  Reconstructed absolute pressure (current + predicted delta):")
    for now_col, target_col, delta_col in wf.PRESSURE_PAIRS:
        reconstructed = test_df[now_col].values + models[delta_col].predict(X_test)
        mae = mean_absolute_error(test_df[target_col], reconstructed)
        print(f"    {target_col:26s} MAE={mae:.1f} Pa")

    # --- Evaluate the blended output, which is what the API returns ------------
    print("\n===== Blended (50% climatology + 50% sensor) test performance =====")
    blend_metrics = {}
    for index_target, clim_name in [
        ("flood_index_target", "flood_index"),
        ("rain_index_target", "rain_index"),
    ]:
        sensor_pred = np.clip(models[index_target].predict(X_test), 0.0, 100.0)
        clim_pred = np.array(
            [wf.climatology_lookup(climatology, clim_name, ts) for ts in test_df["timestamp"]]
        )
        blended = wf.CLIMATE_WEIGHT * clim_pred + wf.SENSOR_WEIGHT * sensor_pred
        truth = test_df[index_target].to_numpy()

        blend_metrics[index_target] = {
            "blended": metrics_for(truth, blended),
            "sensor_only": metrics_for(truth, sensor_pred),
            "climatology_only": metrics_for(truth, clim_pred),
        }
        b, s, c = (blend_metrics[index_target][k]["MAE"] for k in ("blended", "sensor_only", "climatology_only"))
        print(f"  {index_target:20s} MAE  blended={b:.2f}  sensor_only={s:.2f}  climatology_only={c:.2f}")

        # Category agreement of the blended percentage against the truth label.
        to_cat = wf.flood_category if "flood" in index_target else wf.rain_category
        agree = np.mean([to_cat(p) == to_cat(t) for p, t in zip(blended, truth)])
        blend_metrics[index_target]["category_agreement"] = float(agree)
        print(f"  {'':20s} derived-category agreement = {agree:.3f}")

    # --- Responsiveness check: does the sensor half actually move the output? ---
    print("\n===== Sensor responsiveness (same date, dry vs saturated inputs) =====")
    dry_row = train_df.nsmallest(200, "rain_accum_24h")[wf.SENSOR_FEATURE_COLUMNS].mean()
    wet_row = train_df.nlargest(200, "rain_accum_24h")[wf.SENSOR_FEATURE_COLUMNS].mean()
    probe = pd.DataFrame([dry_row, wet_row])
    probe_scaled = pd.DataFrame(
        scaler.transform(probe), columns=wf.SENSOR_FEATURE_COLUMNS
    )
    for index_target in wf.INDEX_TARGETS:
        dry_pct, wet_pct = np.clip(models[index_target].predict(probe_scaled), 0, 100)
        print(f"  {index_target:20s} dry={dry_pct:5.1f}%  saturated={wet_pct:5.1f}%  spread={wet_pct - dry_pct:5.1f}")

    # --- Manifest --------------------------------------------------------------
    manifest = {
        "model_version": MODEL_VERSION,
        "feature_columns": wf.SENSOR_FEATURE_COLUMNS,
        "excluded_time_columns": wf.TIME_COLUMNS,
        "regression_targets": wf.REGRESSION_TARGETS,
        "index_targets": wf.INDEX_TARGETS,
        "pressure_pairs": wf.PRESSURE_PAIRS,
        "blend_weights": {
            "climatology": wf.CLIMATE_WEIGHT,
            "sensor": wf.SENSOR_WEIGHT,
        },
        "index_scales": {
            "rain_index_full_scale_mm": wf.RAIN_INDEX_FULL_SCALE_MM,
            "flood_rain_weight": wf.FLOOD_RAIN_WEIGHT,
            "flood_saturation_weight": wf.FLOOD_SATURATION_WEIGHT,
            "temp_percent_min_c": wf.TEMP_PERCENT_MIN_C,
            "temp_percent_max_c": wf.TEMP_PERCENT_MAX_C,
        },
        "calibration": {
            "rain_adc_dry": wf.RAIN_ADC_DRY,
            "rain_adc_wet": wf.RAIN_ADC_WET,
            "soil_adc_dry": wf.SOIL_ADC_DRY,
            "soil_adc_sat": wf.SOIL_ADC_SAT,
            "rain_rate_full_scale_mm_h": wf.RAIN_RATE_FULL_SCALE_MM_H,
        },
        "category_thresholds": {
            "flood": wf.FLOOD_CATEGORY_THRESHOLDS + [[None, wf.FLOOD_CATEGORY_TOP]],
            "rain": wf.RAIN_CATEGORY_THRESHOLDS + [[None, wf.RAIN_CATEGORY_TOP]],
        },
        "feature_ranges": feature_ranges,
        "dataset": {
            "path": args.data,
            "rows": int(len(df)),
            "interval_min": float(interval_min),
            "first_timestamp": str(df["timestamp"].iloc[0]),
            "last_timestamp": str(df["timestamp"].iloc[-1]),
        },
        "gbm_backend": GBM_BACKEND,
        # Pickled estimators are only guaranteed loadable under the versions they
        # were fitted with; a mismatch is what produced sklearn's
        # InconsistentVersionWarning on the deployed instance.
        "library_versions": {
            "scikit-learn": sklearn_version,
            "lightgbm": getattr(lgb, "__version__", None) if lgb else None,
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "joblib": joblib.__version__,
        },
        "model_metrics": model_metrics,
        "blend_metrics": blend_metrics,
    }
    with open(os.path.join(args.out, "manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=2, default=str)

    print(f"\nArtifacts written to ./{args.out}/")


if __name__ == "__main__":
    main()
