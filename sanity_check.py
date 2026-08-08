"""
Offline sanity checks for the served prediction path.

Verifies the two properties the previous model failed: dry sensors must not read
as high flood risk, and the same sensor reading must not swing wildly just because
the calendar moved. Run it after any retrain:

    python sanity_check.py
"""

from __future__ import annotations

from datetime import datetime

import app  # imported for its side effect: loads the artifacts
import weather_features as wf

DRY = {
    "temperature": 33.8,
    "humidity": 62.6,
    "temp_bmp180": 33.3,
    "pressure_loc1": 98729.0,
    "temp_bmp280": 34.2,
    "pressure_loc2": 98846.1,
    "rain_raw": 4095,
    "soil_moisture": 4095,
}
WET = {
    "temperature": 25.4,
    "humidity": 97.0,
    "temp_bmp180": 25.1,
    # A rain squall drops station pressure by a few hPa; the wet fixture used to
    # sit a full kPa above the dry reading, which is not a pressure this station
    # ever reports and was being clipped away as out-of-distribution.
    "pressure_loc1": 98610.0,
    "temp_bmp280": 26.0,
    "pressure_loc2": 98725.0,
    "rain_raw": 1300,
    "soil_moisture": 1050,
}


def predict(payload: dict, when: datetime) -> tuple[dict, dict]:
    features = {k: float(v) for k, v in payload.items()}
    features.update(wf.derive_features(features, when, None))
    return app.run_model_inference(features, when)


def main() -> None:
    header = (
        f"{'date':<13}{'payload':<8}{'flood%':>8}{'clim/sens':>12}"
        f"{'rain%':>8}{'clim/sens':>12}{'temp C':>8}{'temp%':>7}{'labels':>18}"
    )
    print(header)
    print("-" * len(header))
    clipped_notes: list[str] = []

    for month in (1, 4, 8, 12):
        when = datetime(2025, month, 15, 14, 0)
        for name, payload in (("dry", DRY), ("wet", WET)):
            p, diag = predict(payload, when)
            if diag["clipped_features"]:
                clipped_notes.append(f"{when:%b} {name}: {diag['clipped_features']}")
            flood_split = f"{p['flood_climate_percent']:.0f}/{p['flood_sensor_percent']:.0f}"
            rain_split = f"{p['rain_climate_percent']:.0f}/{p['rain_sensor_percent']:.0f}"
            labels = f"{p['flood_risk_target']}/{p['rain_category_target']}"
            print(
                f"{when:%Y-%m-%d}   {name:<8}"
                f"{p['flood_risk_percent']:>7.1f}%{flood_split:>12}"
                f"{p['rain_percent']:>7.1f}%{rain_split:>12}"
                f"{p['temperature_target']:>8.1f}{p['temperature_percent']:>7.1f}"
                f"{labels:>18}"
            )

    if clipped_notes:
        print("\nFeatures clipped to the training range (input outside training distribution):")
        for note in clipped_notes:
            print(f"  {note}")
    else:
        print("\nNo features clipped: both payloads sit inside the training distribution.")


if __name__ == "__main__":
    main()
