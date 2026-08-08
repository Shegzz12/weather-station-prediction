# Weather station: 48h flood / rainfall / temperature outlook

An ESP32 posts raw sensor readings to a Flask API, which returns a 48-hour
outlook and logs the reading. Predictions are reported as **0-100 percentages**
with a category label kept alongside them.

## How a prediction is produced

Every index is the average of two independent halves:

```
final_percent = 0.5 * climate_percent + 0.5 * sensor_percent
```

- **`climate_percent`** — a climatology lookup keyed on day-of-year only
  (`model_artifacts/climatology.json`). This is the seasonal "what this date
  usually looks like" half.
- **`sensor_percent`** — a gradient-boosted regressor whose inputs are *only*
  sensor readings and features physically derived from them. Calendar columns
  (`hour_sin`, `doy_sin`, `month`, ...) are deliberately excluded, and both
  `train_models.py` and `app.py` refuse to run if any of them reach the model's
  feature list.

Splitting it this way is what makes the 50/50 weighting real: a single model
cannot be constrained to weight the date against the sensors, and the earlier
model collapsed into a pure calendar lookup (dry sensors still read "high" in
August). With the split, dry sensors can never push the sensor half up, and the
date can never move more than half the final number.

Categories are derived from the final percentage, so `flood_risk_target` and
`rain_category_target` stay consistent with the percentages:

| Flood            | Rainfall             |
| ---------------- | -------------------- |
| `< 25` low       | `< 20` dry           |
| `25-60` watch    | `20-45` light        |
| `>= 60` high     | `45-70` moderate     |
|                  | `>= 70` heavy        |

Temperature keeps its actual forecast in °C (`temperature_target`) and also
exposes `temperature_percent`, which is simply the forecast's position on a
15-45 °C scale — a heat level, not a probability.

## Sensor calibration lives in one place

`weather_features.py` is the single source of truth shared by the generator, the
trainer and the API: raw ADC to mm/h rain rate, raw ADC to soil saturation,
index scales, category thresholds and the feature engineering itself. The
firmware therefore sends `rain_raw` / `soil_moisture` as raw 0-4095
`analogRead()` values and never rescales them.

Rain accumulations are integrated in millimetres over real elapsed time from the
telemetry history, so they do not depend on the posting interval. Anything
outside the training range is clipped and reported in
`diagnostics.clipped_features`.

## Regenerating data and models

```bash
pip install -r requirements.txt
python generate_dataset.py            # -> weather_flood_dataset.csv.gz
python train_models.py --clean        # -> model_artifacts/
python sanity_check.py                # dry vs wet payloads across four seasons
```

`generate_dataset.py` simulates multiple full years (multi-day wet/dry regimes,
storms, pre-storm pressure falls, a draining soil bucket) so that season is not
confounded with the labels the way the original single-partial-year dataset was.
`train_models.py` splits chronologically by week with a 48-hour embargo around
every boundary, and reports sensor-only, climatology-only and blended metrics.

Pinned library versions in `requirements.txt` must match the ones recorded in
`model_artifacts/manifest.json` — joblib estimators are only guaranteed to load
under the versions they were fitted with.

## Running the API

```bash
python app.py                                   # dev, port 5000
gunicorn app:app --bind 0.0.0.0:$PORT           # production
```

`POST /api/telemetry` takes `temperature`, `humidity`, `temp_bmp180`,
`pressure_loc1`, `temp_bmp280`, `pressure_loc2`, `rain_raw`, `soil_moisture`,
returns `predictions_48h` plus the derived features and blend diagnostics, and
appends the row to `data/telemetry_history.csv`. Also available:
`GET /api/health`, `/api/model`, `/api/latest`, `/api/history`, `/api/export`
and `POST /api/reset`. The dashboard is served from `/`.

## Firmware

`firmware/weather_station/weather_station.ino`. Copy
`arduino_secrets.h.example` to `arduino_secrets.h` and fill in your WiFi
credentials; that file is git-ignored.
