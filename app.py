# ==============================================================================
# Production Flask API — ESP32 Sensor Telemetry & 48h Weather/Flood Forecast
# ==============================================================================

import os
import json
import math
import joblib
import threading
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify, send_from_directory, send_file
from flask_cors import CORS

# ------------------------------------------------------------------------------
# 1. Initialization & Configuration
# ------------------------------------------------------------------------------
app = Flask(__name__, static_folder="static", static_url_path="")
CORS(app)  # Enables Cross-Origin Resource Sharing for Frontend Integration

ARTIFACTS_DIR = "model_artifacts"
DATA_DIR = "data"
CSV_FILE_PATH = os.path.join(DATA_DIR, "telemetry_history.csv")

# Thread lock to prevent race conditions during CSV read/write operations
csv_lock = threading.Lock()

os.makedirs(DATA_DIR, exist_ok=True)

# ------------------------------------------------------------------------------
# 2. Load Model Artifacts & Manifest
# ------------------------------------------------------------------------------
print("⏳ Loading machine learning artifacts...")

try:
    with open(os.path.join(ARTIFACTS_DIR, "manifest.json"), "r") as f:
        manifest = json.load(f)

    scaler = joblib.load(os.path.join(ARTIFACTS_DIR, "feature_scaler.joblib"))
    FEATURE_COLS = manifest["feature_columns"]
    HEAVY_THRESHOLD = manifest.get("heavy_rain_threshold", 0.20)

    regression_models = {
        target: joblib.load(os.path.join(ARTIFACTS_DIR, f"model_{target}.joblib"))
        for target in manifest["regression_targets"]
    }

    classification_models = {
        target: joblib.load(os.path.join(ARTIFACTS_DIR, f"model_{target}.joblib"))
        for target in manifest["classification_targets"]
    }

    classification_encoders = {
        target: joblib.load(os.path.join(ARTIFACTS_DIR, f"encoder_{target}.joblib"))
        for target in manifest["classification_targets"]
    }

    print("✅ All artifacts successfully loaded into memory!")

except Exception as e:
    print(f"❌ Error loading model artifacts from '{ARTIFACTS_DIR}': {e}")
    raise e


# ------------------------------------------------------------------------------
# 3. CSV Storage Engine
# ------------------------------------------------------------------------------
# All columns to be stored persistently in CSV
LOG_COLUMNS = (
    ["timestamp"]
    + FEATURE_COLS
    + [
        "temperature_target",
        "humidity_target",
        "pressure_loc1_target",
        "pressure_loc2_target",
        "rain_category_target",
        "flood_risk_target",
    ]
)

def init_csv_storage():
    """Initializes the CSV log file with headers if it does not exist."""
    with csv_lock:
        if not os.path.exists(CSV_FILE_PATH):
            df_empty = pd.DataFrame(columns=LOG_COLUMNS)
            df_empty.to_csv(CSV_FILE_PATH, index=False)
            print(f"📁 Created initial storage log: {CSV_FILE_PATH}")

init_csv_storage()


def sanitize_records(df: pd.DataFrame) -> list:
    """
    Converts a DataFrame to a list of JSON-safe dicts.
    pandas leaves NaN for any missing/misaligned cell, and Python's json module
    will happily emit the literal token `NaN`, which is NOT valid JSON — browsers'
    res.json() throws a SyntaxError on it, which was silently breaking the
    frontend's history chart and raw log table. Replacing NaN with None here
    makes it serialize as JSON `null`, which every client can parse safely.
    """
    return json.loads(df.where(pd.notnull(df), None).to_json(orient="records"))


def sanitize_record(d: dict) -> dict:
    """Same NaN->None sanitization, but for a single dict (e.g. df.iloc[-1].to_dict())."""
    clean = {}
    for k, v in d.items():
        if isinstance(v, float) and math.isnan(v):
            clean[k] = None
        else:
            clean[k] = v
    return clean


# ------------------------------------------------------------------------------
# 4. Feature Engineering Helpers (Real-Time Calculation)
# ------------------------------------------------------------------------------
def compute_time_features(dt: datetime) -> dict:
    """Computes cyclical time features from datetime."""
    hour = dt.hour + dt.minute / 60.0
    doy = dt.timetuple().tm_yday

    return {
        "hour_sin": math.sin(2 * math.pi * hour / 24.0),
        "hour_cos": math.cos(2 * math.pi * hour / 24.0),
        "doy_sin": math.sin(2 * math.pi * doy / 365.25),
        "doy_cos": math.cos(2 * math.pi * doy / 365.25),
        "month": dt.month,
    }


def compute_derived_features(raw_data: dict, current_time: datetime) -> dict:
    """
    Computes rolling accumulation and trend features by reading recent history from CSV.
    Falls back to sensible defaults if insufficient historical records exist.
    """
    p1_current = raw_data["pressure_loc1"]
    p2_current = raw_data["pressure_loc2"]
    h_current = raw_data["humidity"]
    rain_raw = raw_data.get("rain_raw", 0.0)

    # Defaults for cold start (empty CSV or new system initialization)
    p1_trend_3h, p1_trend_6h = 0.0, 0.0
    p2_trend_3h, p2_trend_6h = 0.0, 0.0
    humidity_trend_3h = 0.0
    rain_accum_6h = rain_raw
    rain_accum_24h = rain_raw

    with csv_lock:
        if os.path.exists(CSV_FILE_PATH) and os.path.getsize(CSV_FILE_PATH) > 0:
            try:
                df_hist = pd.read_csv(CSV_FILE_PATH)
                if not df_hist.empty:
                    df_hist["dt"] = pd.to_datetime(df_hist["timestamp"])

                    # 3-Hour Lookback
                    t_3h = current_time - timedelta(hours=3)
                    df_3h = df_hist[df_hist["dt"] >= t_3h]
                    if not df_3h.empty:
                        oldest_3h = df_3h.iloc[0]
                        p1_trend_3h = p1_current - oldest_3h["pressure_loc1"]
                        p2_trend_3h = p2_current - oldest_3h["pressure_loc2"]
                        humidity_trend_3h = h_current - oldest_3h["humidity"]

                    # 6-Hour Lookback
                    t_6h = current_time - timedelta(hours=6)
                    df_6h = df_hist[df_hist["dt"] >= t_6h]
                    if not df_6h.empty:
                        oldest_6h = df_6h.iloc[0]
                        p1_trend_6h = p1_current - oldest_6h["pressure_loc1"]
                        p2_trend_6h = p2_current - oldest_6h["pressure_loc2"]
                        rain_accum_6h = df_6h["rain_raw"].sum() + rain_raw

                    # 24-Hour Lookback
                    t_24h = current_time - timedelta(hours=24)
                    df_24h = df_hist[df_hist["dt"] >= t_24h]
                    if not df_24h.empty:
                        rain_accum_24h = df_24h["rain_raw"].sum() + rain_raw

            except Exception as ex:
                print(f"⚠️ Warning during trend calculation: {ex}")

    return {
        "pressure_loc1_trend_3h": float(p1_trend_3h),
        "pressure_loc1_trend_6h": float(p1_trend_6h),
        "pressure_loc2_trend_3h": float(p2_trend_3h),
        "pressure_loc2_trend_6h": float(p2_trend_6h),
        "pressure_diff": float(p1_current - p2_current),
        "humidity_trend_3h": float(humidity_trend_3h),
        "rain_rate": float(raw_data.get("rain_rate", rain_raw)),
        "rain_accum_6h": float(rain_accum_6h),
        "rain_accum_24h": float(rain_accum_24h),
    }


# ------------------------------------------------------------------------------
# 5. Core Model Inference Engine
# ------------------------------------------------------------------------------
def run_model_inference(feature_dataframe: pd.DataFrame) -> dict:
    """Runs input feature vectors through model pipeline."""
    # Ensure strict feature ordering matching scaler training
    X_input = feature_dataframe[FEATURE_COLS].copy()
    X_scaled = pd.DataFrame(
        scaler.transform(X_input), columns=FEATURE_COLS, index=X_input.index
    )

    results = {}

    # Direct Regressions (Temperature & Humidity)
    for target in manifest["regression_targets"]:
        if target in ["temperature_target", "humidity_target"]:
            results[target] = float(regression_models[target].predict(X_scaled)[0])

    # Pressure Delta Reconstruction
    for now_col, target_col, delta_col in manifest["pressure_pairs"]:
        pred_delta = regression_models[delta_col].predict(X_scaled)[0]
        now_val = feature_dataframe[now_col].values[0]
        results[target_col] = float(now_val + pred_delta)

    # Classification Targets with Calibrated Threshold
    for target in manifest["classification_targets"]:
        model = classification_models[target]
        encoder = classification_encoders[target]
        probs = model.predict_proba(X_scaled)

        preds_enc = np.argmax(probs, axis=1)

        if target == "rain_category_target" and "heavy" in encoder.classes_:
            heavy_idx = np.where(encoder.classes_ == "heavy")[0][0]
            if probs[0, heavy_idx] >= HEAVY_THRESHOLD:
                preds_enc[0] = heavy_idx

        results[target] = str(encoder.inverse_transform(preds_enc)[0])

    return results


# ------------------------------------------------------------------------------
# 6. Frontend Route
# ------------------------------------------------------------------------------
@app.route("/", methods=["GET"])
def serve_dashboard():
    """Serves the single-file dashboard from static/index.html."""
    return send_from_directory(app.static_folder, "index.html")


# ------------------------------------------------------------------------------
# 7. API Routes / Endpoints
# ------------------------------------------------------------------------------

@app.route("/api/health", methods=["GET"])
def health_check():
    """System health check endpoint."""
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "model_version": "v6_calibrated",
        "artifacts_directory": ARTIFACTS_DIR
    }), 200


@app.route("/api/telemetry", methods=["POST"])
def receive_telemetry():
    """
    POST Endpoint for ESP32.
    Receives raw sensor readings, computes engineered features, executes model inference,
    logs row to CSV, and returns real-time prediction output.
    """
    try:
        data = request.get_json(force=True)
        if not data:
            return jsonify({"error": "Invalid or missing JSON payload"}), 400

        # Required fields from ESP32
        required_fields = [
            "temperature", "humidity", "temp_bmp180", "pressure_loc1",
            "temp_bmp280", "pressure_loc2", "rain_raw", "soil_moisture"
        ]
        missing = [field for field in required_fields if field not in data]
        if missing:
            return jsonify({"error": f"Missing required sensor fields: {missing}"}), 400

        # Current Timestamp
        now = datetime.now()
        timestamp_str = now.strftime("%Y-%m-%d %H:%M:%S")

        # Compute Engineered Features
        time_feats = compute_time_features(now)
        derived_feats = compute_derived_features(data, now)

        # Build complete feature dictionary
        full_feature_row = {}
        full_feature_row.update(time_feats)
        for k in required_fields:
            full_feature_row[k] = float(data[k])
        full_feature_row.update(derived_feats)

        df_features = pd.DataFrame([full_feature_row])

        # Execute Predictions
        predictions = run_model_inference(df_features)

        # Assemble full log record
        log_record = {"timestamp": timestamp_str}
        log_record.update(full_feature_row)
        log_record.update(predictions)

        # Append to CSV storage
        with csv_lock:
            df_log = pd.DataFrame([log_record])
            df_log[LOG_COLUMNS].to_csv(CSV_FILE_PATH, mode="a", header=False, index=False)

        print(f"✅ Telemetry logged at {timestamp_str} | Rain: {predictions['rain_category_target'].upper()} | Flood Risk: {predictions['flood_risk_target'].upper()}")

        return jsonify({
            "status": "success",
            "timestamp": timestamp_str,
            "received_sensor_data": data,
            "calculated_features": derived_feats,
            "predictions_48h": predictions
        }), 201

    except Exception as e:
        print(f"❌ Exception in /api/telemetry: {str(e)}")
        return jsonify({"error": "Internal server processing error", "details": str(e)}), 500


@app.route("/api/latest", methods=["GET"])
def get_latest():
    """Retrieves the single most recent sensor reading and prediction record."""
    try:
        with csv_lock:
            if not os.path.exists(CSV_FILE_PATH):
                return jsonify({"error": "No historical log available yet"}), 444

            df = pd.read_csv(CSV_FILE_PATH)
            if df.empty:
                return jsonify({"message": "Database log is empty"}), 200

            latest_record = sanitize_record(df.iloc[-1].to_dict())
            return jsonify(latest_record), 200
    except Exception as e:
        print(f"❌ Exception in /api/latest: {str(e)}")
        return jsonify({"error": "Failed to read latest record", "details": str(e)}), 500


@app.route("/api/history", methods=["GET"])
def get_history():
    """
    Retrieves historical records for time-series charts on the frontend dashboard.
    Supports query parameters:
      - limit: int (default: 100, max: 2000)
      - order: 'desc' or 'asc' (default: 'desc')
    """
    limit = request.args.get("limit", default=100, type=int)
    order = request.args.get("order", default="desc", type=str).lower()

    limit = min(max(1, limit), 2000)  # Bound limit between 1 and 2000

    try:
        with csv_lock:
            if not os.path.exists(CSV_FILE_PATH):
                return jsonify({"count": 0, "order": order, "data": []}), 200

            df = pd.read_csv(CSV_FILE_PATH)
            if df.empty:
                return jsonify({"count": 0, "order": order, "data": []}), 200

            if order == "desc":
                df_sliced = df.tail(limit).iloc[::-1]
            else:
                df_sliced = df.tail(limit)

            records = sanitize_records(df_sliced)
            return jsonify({
                "count": len(records),
                "order": order,
                "data": records
            }), 200
    except Exception as e:
        print(f"❌ Exception in /api/history: {str(e)}")
        return jsonify({"error": "Failed to read history", "details": str(e)}), 500


@app.route("/api/export", methods=["GET"])
def export_csv():
    """
    Downloads the full raw telemetry_history.csv file as-is, for offline
    analysis, backup, or re-training. Streams the file directly rather than
    going through pandas/JSON, so it works even if a row somewhere has a
    malformed value that would otherwise break the JSON-based endpoints.
    """
    with csv_lock:
        if not os.path.exists(CSV_FILE_PATH):
            return jsonify({"error": "No historical log available yet"}), 404

        timestamp_suffix = datetime.now().strftime("%Y%m%d_%H%M%S")
        download_name = f"telemetry_history_{timestamp_suffix}.csv"

        return send_file(
            CSV_FILE_PATH,
            mimetype="text/csv",
            as_attachment=True,
            download_name=download_name,
        )


@app.route("/api/reset", methods=["POST"])
def reset_csv():
    """
    Wipes the live telemetry log so new readings start accumulating from
    scratch (e.g. after test data or a schema change has polluted the
    rolling 6h/24h trend features). Does NOT delete data outright — the
    existing CSV is renamed into a timestamped backup first, so nothing
    is unrecoverably lost even on an accidental click.

    Requires a JSON body {"confirm": true} to guard against this being
    triggered by anything other than a deliberate frontend action (a bare
    GET-able endpoint would be too easy to hit by accident via a stray link
    or crawler).
    """
    payload = request.get_json(silent=True) or {}
    if payload.get("confirm") is not True:
        return jsonify({"error": "Reset requires JSON body {\"confirm\": true}"}), 400

    with csv_lock:
        backup_name = None
        if os.path.exists(CSV_FILE_PATH) and os.path.getsize(CSV_FILE_PATH) > 0:
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_name = f"telemetry_history_backup_{stamp}.csv"
            backup_path = os.path.join(DATA_DIR, backup_name)
            os.rename(CSV_FILE_PATH, backup_path)
            print(f"🗄️  Backed up existing log to: {backup_path}")

        df_empty = pd.DataFrame(columns=LOG_COLUMNS)
        df_empty.to_csv(CSV_FILE_PATH, index=False)
        print(f"🧹 Reset {CSV_FILE_PATH} — logging starts fresh from here.")

    return jsonify({
        "status": "reset",
        "backup_created": backup_name,
        "message": (
            f"Live log cleared. Prior data preserved as '{backup_name}' in {DATA_DIR}/."
            if backup_name else "Live log was already empty; nothing to back up."
        )
    }), 200


# ------------------------------------------------------------------------------
# 8. Server Start
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    # Host on 0.0.0.0 so ESP32 on local network can communicate with Flask host
    app.run(host="0.0.0.0", port=5000, debug=False)
