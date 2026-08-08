# ==============================================================================
# Production Flask API — ESP32 Sensor Telemetry & 48h Weather/Flood Forecast
#
# Every index this API returns is a 0-100 percentage built as
#
#     final = CLIMATE_WEIGHT * climatology(date) + SENSOR_WEIGHT * model(sensors)
#
# with both weights fixed at 0.50 (see weather_features.py). The models
# themselves receive no calendar features at all, so the season can influence the
# answer only through the climatology term and can never outvote the sensors.
# The legacy string categories are still returned, derived from the final
# percentage, so existing clients keep working.
#
# LOGGING: every incoming request is logged before routing, and every
# response's status code is logged after the route runs, so Render's log
# stream shows full request/response traffic — not just requests that
# happen to hit our own exception handlers. See section 1b below.
# ==============================================================================

import json
import logging
import math
import os
import sys
import threading
import traceback
from datetime import datetime, timedelta

import joblib
import pandas as pd
from flask import Flask, jsonify, request, send_file, send_from_directory
from flask_cors import CORS

import weather_features as wf

# ------------------------------------------------------------------------------
# 1. Initialization & Configuration
# ------------------------------------------------------------------------------
app = Flask(__name__, static_folder="static", static_url_path="")
CORS(app)  # Enables Cross-Origin Resource Sharing for Frontend Integration

ARTIFACTS_DIR = os.environ.get("ARTIFACTS_DIR", "model_artifacts")
DATA_DIR = "data"
CSV_FILE_PATH = os.path.join(DATA_DIR, "telemetry_history.csv")

# Feature engineering needs at most 24h of history; read a little more than that.
HISTORY_WINDOW_HOURS = 26

csv_lock = threading.Lock()
os.makedirs(DATA_DIR, exist_ok=True)

# ------------------------------------------------------------------------------
# 1b. Logging — request/response visibility on Render
# ------------------------------------------------------------------------------
# Render captures whatever the process writes to stdout/stderr as the service
# log, so a plain stream handler on stdout is enough for these lines to show
# up live in the Render dashboard's Logs tab.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    stream=sys.stdout,
)
log = logging.getLogger("telemetry_api")


@app.before_request
def log_incoming_request():
    """
    Logs every incoming request before it reaches a route handler, so the
    Render log stream shows *all* traffic — including requests that get
    rejected before our own route logic runs — not only the ones that reach
    an explicit log line or exception handler inside a view function.
    """
    body_preview = ""
    if request.method in ("POST", "PUT", "PATCH"):
        try:
            # cache=True is required here: it lets Werkzeug buffer the body
            # so request.get_json()/request.get_data() inside the route can
            # still read it afterward. Without cache=True the stream would
            # already be consumed and every POST would 400 on missing JSON.
            raw = request.get_data(cache=True, as_text=True)
            body_preview = f" body={raw[:500]!r}" if raw else " body=<empty>"
        except Exception as exc:  # pragma: no cover - defensive only
            body_preview = f" body=<unreadable: {exc}>"

    log.info(
        "--> %s %s from %s%s",
        request.method,
        request.path,
        request.headers.get("X-Forwarded-For", request.remote_addr),
        body_preview,
    )


@app.after_request
def log_response_status(response):
    """Logs the outcome of every request, matched against the line above."""
    log.info("<-- %s %s -> %s", request.method, request.path, response.status_code)
    return response


@app.errorhandler(Exception)
def log_unhandled_exception(exc):
    """
    Safety net for anything that escapes a route's own try/except (e.g. a
    routing error, or a bug in a handler that doesn't wrap its body). Logs
    the full traceback through the same logger — Render's log stream is the
    single place to look, instead of some errors surfacing only via
    traceback.print_exc() in a different format.
    """
    log.error("Unhandled exception on %s %s:\n%s", request.method, request.path,
               traceback.format_exc())
    return jsonify({"error": "Internal server processing error"}), 500


# ------------------------------------------------------------------------------
# 2. Load Model Artifacts & Manifest
# ------------------------------------------------------------------------------
log.info("Loading machine learning artifacts...")

with open(os.path.join(ARTIFACTS_DIR, "manifest.json"), "r") as fh:
    manifest = json.load(fh)

with open(os.path.join(ARTIFACTS_DIR, "climatology.json"), "r") as fh:
    climatology = json.load(fh)

scaler = joblib.load(os.path.join(ARTIFACTS_DIR, "feature_scaler.joblib"))
FEATURE_COLS = manifest["feature_columns"]
FEATURE_RANGES = manifest["feature_ranges"]
MODEL_VERSION = manifest["model_version"]

models = {
    target: joblib.load(os.path.join(ARTIFACTS_DIR, f"model_{target}.joblib"))
    for target in manifest["regression_targets"] + manifest["index_targets"]
}

leaked = [c for c in wf.TIME_COLUMNS if c in FEATURE_COLS]
if leaked:
    raise RuntimeError(
        f"Artifacts in '{ARTIFACTS_DIR}' were trained with calendar features {leaked}. "
        "Retrain with train_models.py — the calendar must enter only via the climatology."
    )

log.info("Artifacts loaded (%s, %d sensor features)", MODEL_VERSION, len(FEATURE_COLS))

# ------------------------------------------------------------------------------
# 3. CSV Storage Engine
# ------------------------------------------------------------------------------
PREDICTION_COLUMNS = [
    "temperature_target",
    "humidity_target",
    "pressure_loc1_target",
    "pressure_loc2_target",
    "flood_risk_percent",
    "flood_sensor_percent",
    "flood_climate_percent",
    "rain_percent",
    "rain_sensor_percent",
    "rain_climate_percent",
    "temperature_percent",
    "rain_category_target",
    "flood_risk_target",
]

LOG_COLUMNS = (
    ["timestamp"] + wf.TIME_COLUMNS + wf.SENSOR_FEATURE_COLUMNS + PREDICTION_COLUMNS
)


def init_csv_storage():
    """
    Create the log file, or move an incompatible one aside.

    The v7 schema adds the percentage columns, so a log written by an older
    build cannot be appended to — mixing the two would silently misalign every
    column. The old file is renamed, never deleted.
    """
    with csv_lock:
        if os.path.exists(CSV_FILE_PATH) and os.path.getsize(CSV_FILE_PATH) > 0:
            existing = pd.read_csv(CSV_FILE_PATH, nrows=0).columns.tolist()
            if existing == LOG_COLUMNS:
                return
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup = os.path.join(DATA_DIR, f"telemetry_history_pre_v7_{stamp}.csv")
            os.rename(CSV_FILE_PATH, backup)
            log.warning("Log schema changed; previous log preserved as %s", backup)

        pd.DataFrame(columns=LOG_COLUMNS).to_csv(CSV_FILE_PATH, index=False)
        log.info("Created storage log: %s", CSV_FILE_PATH)


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
    """Same NaN->None sanitization, but for a single dict."""
    return {
        k: (None if isinstance(v, float) and math.isnan(v) else v) for k, v in d.items()
    }


def read_recent_history(now: datetime) -> pd.DataFrame:
    """Recent raw readings used to build trend and accumulation features."""
    with csv_lock:
        if not os.path.exists(CSV_FILE_PATH) or os.path.getsize(CSV_FILE_PATH) == 0:
            return pd.DataFrame()
        try:
            df = pd.read_csv(
                CSV_FILE_PATH, usecols=["timestamp"] + wf.RAW_SENSOR_COLUMNS
            )
        except (ValueError, pd.errors.EmptyDataError) as exc:
            log.warning("Could not read history for trend features: %s", exc)
            return pd.DataFrame()

    if df.empty:
        return df
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    cutoff = now - timedelta(hours=HISTORY_WINDOW_HOURS)
    return df[df["timestamp"] >= cutoff].dropna(subset=["timestamp"]).sort_values("timestamp")


# ------------------------------------------------------------------------------
# 4. Core Model Inference Engine
# ------------------------------------------------------------------------------
def run_model_inference(feature_row: dict, now: datetime) -> tuple[dict, dict]:
    """
    Predict every 48h target for one reading.

    Returns (predictions, diagnostics). `diagnostics` reports the separate
    climatology and sensor halves of each index plus any feature that had to be
    clipped back into the training range, so a bad reading is visible in the
    response instead of silently distorting the answer.
    """
    clipped_row, clipped = wf.clip_to_training_range(feature_row, FEATURE_RANGES)
    if clipped:
        log.warning("Clipped out-of-range features: %s", clipped)

    frame = pd.DataFrame([clipped_row])[FEATURE_COLS]
    scaled = pd.DataFrame(scaler.transform(frame), columns=FEATURE_COLS)

    predictions = {
        "temperature_target": float(models["temperature_target"].predict(scaled)[0]),
        "humidity_target": float(models["humidity_target"].predict(scaled)[0]),
    }

    for now_col, target_col, delta_col in manifest["pressure_pairs"]:
        delta = float(models[delta_col].predict(scaled)[0])
        predictions[target_col] = float(feature_row[now_col]) + delta

    diagnostics = {"clipped_features": clipped, "components": {}}

    for index_target, clim_name, prefix, category_key, categoriser in [
        ("flood_index_target", "flood_index", "flood_risk", "flood_risk_target", wf.flood_category),
        ("rain_index_target", "rain_index", "rain", "rain_category_target", wf.rain_category),
    ]:
        sensor_pct = wf.clamp(float(models[index_target].predict(scaled)[0]), 0.0, 100.0)
        climate_pct = wf.clamp(
            wf.climatology_lookup(climatology, clim_name, now), 0.0, 100.0
        )
        blended = wf.blend_percent(climate_pct, sensor_pct)

        predictions[f"{prefix}_percent"] = round(blended, 1)
        predictions[f"{prefix.replace('_risk', '')}_sensor_percent"] = round(sensor_pct, 1)
        predictions[f"{prefix.replace('_risk', '')}_climate_percent"] = round(climate_pct, 1)
        predictions[category_key] = categoriser(blended)
        diagnostics["components"][prefix] = {
            "climatology_percent": round(climate_pct, 1),
            "sensor_percent": round(sensor_pct, 1),
            "weights": manifest["blend_weights"],
        }

    predictions["temperature_percent"] = round(
        wf.temperature_to_percent(predictions["temperature_target"]), 1
    )

    return predictions, diagnostics


# ------------------------------------------------------------------------------
# 5. Frontend Route
# ------------------------------------------------------------------------------
@app.route("/", methods=["GET"])
def serve_dashboard():
    """Serves the single-file dashboard from static/index.html."""
    return send_from_directory(app.static_folder, "index.html")


# ------------------------------------------------------------------------------
# 6. API Routes / Endpoints
# ------------------------------------------------------------------------------
@app.route("/api/health", methods=["GET"])
def health_check():
    """System health check endpoint."""
    return (
        jsonify(
            {
                "status": "healthy",
                "timestamp": datetime.now().isoformat(),
                "model_version": MODEL_VERSION,
                "artifacts_directory": ARTIFACTS_DIR,
                "blend_weights": manifest["blend_weights"],
                "feature_count": len(FEATURE_COLS),
            }
        ),
        200,
    )


@app.route("/api/model", methods=["GET"])
def model_info():
    """
    Exposes the manifest so the scales behind the percentages are inspectable
    (blend weights, index full-scale values, sensor calibration, test metrics).
    """
    return jsonify(manifest), 200


@app.route("/api/telemetry", methods=["POST"])
def receive_telemetry():
    """
    POST endpoint for the ESP32: raw sensor readings in, 48h percentages out.
    Also appends the reading and its prediction to the CSV log.
    """
    try:
        data = request.get_json(force=True)
        if not data:
            log.warning("Rejected telemetry: invalid or missing JSON payload")
            return jsonify({"error": "Invalid or missing JSON payload"}), 400

        missing = [field for field in wf.RAW_SENSOR_COLUMNS if field not in data]
        if missing:
            log.warning(
                "Rejected telemetry: missing required sensor fields %s (got keys: %s)",
                missing, list(data.keys()),
            )
            return jsonify({"error": f"Missing required sensor fields: {missing}"}), 400

        now = datetime.now()
        timestamp_str = now.strftime("%Y-%m-%d %H:%M:%S")

        raw = {field: float(data[field]) for field in wf.RAW_SENSOR_COLUMNS}
        history = read_recent_history(now)
        derived = wf.derive_features(raw, now, history)

        feature_row = {**raw, **derived}
        predictions, diagnostics = run_model_inference(feature_row, now)

        log_record = {"timestamp": timestamp_str}
        log_record.update(wf.time_features(now))
        log_record.update(feature_row)
        log_record.update(
            {k: v for k, v in predictions.items() if k in PREDICTION_COLUMNS}
        )

        with csv_lock:
            pd.DataFrame([log_record])[LOG_COLUMNS].to_csv(
                CSV_FILE_PATH, mode="a", header=False, index=False
            )

        log.info(
            "%s | flood %.0f%% (%s) | rain %.0f%% (%s) | temp %.1fC",
            timestamp_str,
            predictions["flood_risk_percent"], predictions["flood_risk_target"],
            predictions["rain_percent"], predictions["rain_category_target"],
            predictions["temperature_target"],
        )

        return (
            jsonify(
                {
                    "status": "success",
                    "timestamp": timestamp_str,
                    "received_sensor_data": data,
                    "calculated_features": derived,
                    "predictions_48h": predictions,
                    "diagnostics": diagnostics,
                }
            ),
            201,
        )

    except Exception:
        # Full traceback goes to the log (visible in Render); the response
        # stays generic because this endpoint is unauthenticated and
        # exception text carries filesystem paths and internal column names.
        log.error("Exception in /api/telemetry:\n%s", traceback.format_exc())
        return jsonify({"error": "Internal server processing error"}), 500


@app.route("/api/latest", methods=["GET"])
def get_latest():
    """Retrieves the single most recent sensor reading and prediction record."""
    try:
        with csv_lock:
            if not os.path.exists(CSV_FILE_PATH):
                log.warning("GET /api/latest: no log file yet")
                return jsonify({"error": "No historical log available yet"}), 404

            df = pd.read_csv(CSV_FILE_PATH)
            if df.empty:
                log.info("GET /api/latest: log file empty")
                return jsonify({"message": "Database log is empty"}), 200

            return jsonify(sanitize_record(df.iloc[-1].to_dict())), 200
    except Exception:
        log.error("Exception in /api/latest:\n%s", traceback.format_exc())
        return jsonify({"error": "Failed to read latest record"}), 500


@app.route("/api/history", methods=["GET"])
def get_history():
    """
    Retrieves historical records for time-series charts on the frontend dashboard.
    Supports query parameters:
      - limit: int (default: 100, max: 2000)
      - order: 'desc' or 'asc' (default: 'desc')
    """
    limit = min(max(1, request.args.get("limit", default=100, type=int)), 2000)
    order = request.args.get("order", default="desc", type=str).lower()

    try:
        with csv_lock:
            if not os.path.exists(CSV_FILE_PATH):
                return jsonify({"count": 0, "order": order, "data": []}), 200

            df = pd.read_csv(CSV_FILE_PATH)
            if df.empty:
                return jsonify({"count": 0, "order": order, "data": []}), 200

            sliced = df.tail(limit)
            if order == "desc":
                sliced = sliced.iloc[::-1]

            records = sanitize_records(sliced)
            return jsonify({"count": len(records), "order": order, "data": records}), 200
    except Exception:
        log.error("Exception in /api/history:\n%s", traceback.format_exc())
        return jsonify({"error": "Failed to read history"}), 500


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
            log.warning("GET /api/export: no log file yet")
            return jsonify({"error": "No historical log available yet"}), 404

        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return send_file(
            CSV_FILE_PATH,
            mimetype="text/csv",
            as_attachment=True,
            download_name=f"telemetry_history_{stamp}.csv",
        )


@app.route("/api/reset", methods=["POST"])
def reset_csv():
    """
    Wipes the live telemetry log so new readings start accumulating from
    scratch (e.g. after test data or a schema change has polluted the
    rolling 6h/24h trend features). Does NOT delete data outright — the
    existing CSV is renamed into a timestamped backup first.

    Requires a JSON body {"confirm": true} so a stray link or crawler cannot
    trigger it.
    """
    payload = request.get_json(silent=True) or {}
    if payload.get("confirm") is not True:
        log.warning("Reset rejected: missing/false confirm flag (payload=%s)", payload)
        return jsonify({"error": 'Reset requires JSON body {"confirm": true}'}), 400

    with csv_lock:
        backup_name = None
        if os.path.exists(CSV_FILE_PATH) and os.path.getsize(CSV_FILE_PATH) > 0:
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_name = f"telemetry_history_backup_{stamp}.csv"
            os.rename(CSV_FILE_PATH, os.path.join(DATA_DIR, backup_name))
            log.warning("Backed up existing log to: %s", backup_name)

        pd.DataFrame(columns=LOG_COLUMNS).to_csv(CSV_FILE_PATH, index=False)
        log.info("Reset %s — logging starts fresh from here.", CSV_FILE_PATH)

    return (
        jsonify(
            {
                "status": "reset",
                "backup_created": backup_name,
                "message": (
                    f"Live log cleared. Prior data preserved as '{backup_name}' in {DATA_DIR}/."
                    if backup_name
                    else "Live log was already empty; nothing to back up."
                ),
            }
        ),
        200,
    )


# ------------------------------------------------------------------------------
# 7. Server Start
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    # Host on 0.0.0.0 so ESP32 on local network can communicate with Flask host
    app.run(host="0.0.0.0", port=5000, debug=False)
