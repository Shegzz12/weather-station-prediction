"""
Synthetic dataset generator for the weather station's 48h-ahead models.

Why this exists
---------------
The previous `weather_flood_dataset.csv` had two fatal properties:

1. It covered a single partial year (Jan-Jul 2025) in which the flood label was a
   monotone function of the month (Jan/Feb all "low", Jun/Jul all "high"). Any
   learner given day-of-year features simply memorised the calendar, so the
   sensors had no influence on the output.
2. Its feature scales were mutually inconsistent (pressures up to 232 kPa,
   `rain_accum_6h` in millimetres while the API fed it a sum of raw ADC counts).

This generator fixes both: it simulates several full years, so every season is
observed under a range of conditions and season alone cannot determine the label,
and it derives every feature through `weather_features.py` — the same code the
API uses at inference time.

The simulation is a simple physical model of a humid-tropical (Nigeria-like)
station: bimodal wet seasons, storms arriving after a pressure fall, a soil
bucket that fills with rain and drains/evaporates between events.

Usage:
    python generate_dataset.py [--years 4] [--interval-min 30] [--out weather_flood_dataset.csv.gz]

The default output is gzip-compressed (pandas reads it transparently); at 4 years
of 30-minute samples the plain CSV is ~16 MB, which has no business being in git.
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

import weather_features as wf

# Station climate constants (humid tropical, two rainy peaks per year).
#
# The pressure base and temperature range are set to match what this rig actually
# reports (BMP180 ~98.72 kPa, BMP280 ~98.85 kPa, air 31-35 C in August). Training
# on a sea-level 101 kPa distribution would put every real reading outside the
# training range, where the models can only extrapolate. Re-measure and adjust
# these if the station moves or the sensors are recalibrated.
TEMP_ANNUAL_MEAN_C = 29.5
TEMP_ANNUAL_AMPLITUDE_C = 3.5
TEMP_DIURNAL_AMPLITUDE_C = 5.5
HUMIDITY_DRY_SEASON = 45.0
HUMIDITY_WET_SEASON = 86.0
PRESSURE_BASE_PA = 98_800.0
STATION2_PRESSURE_OFFSET_PA = 117.0

REGIME_TIMESCALE_H = 168.0     # ~1-week persistence of wet/dry spells

SOIL_CAPACITY_MM = 90.0        # bucket depth: rain beyond this runs off
SOIL_DRAIN_MM_PER_DAY = 7.0    # drainage + evapotranspiration in dry weather

FORECAST_HORIZON_H = 48


def seasonal_wetness(doy: np.ndarray) -> np.ndarray:
    """
    0..1 seasonal rainfall climatology with peaks in June and September and a
    pronounced dry season around December-February.
    """
    phase = 2 * np.pi * (doy / 365.25)
    primary = np.cos(phase - 2 * np.pi * (170 / 365.25))    # mid-June peak
    secondary = np.cos(2 * (phase - 2 * np.pi * (255 / 365.25)))  # September peak
    raw = 0.62 * primary + 0.38 * secondary
    return np.clip((raw + 1.0) / 2.0, 0.0, 1.0) ** 1.6


def _ou_process(rng, periods: int, step_h: float, tau_h: float) -> np.ndarray:
    """Unit-variance Ornstein-Uhlenbeck series with an `tau_h`-hour memory."""
    decay = np.exp(-step_h / tau_h)
    shocks = rng.normal(0.0, 1.0, periods)
    out = np.empty(periods)
    state = 0.0
    for k in range(periods):
        state = decay * state + np.sqrt(1 - decay**2) * shocks[k]
        out[k] = state
    return out


def simulate(years: int, interval_min: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    step_h = interval_min / 60.0
    start = pd.Timestamp(f"{2025 - years}-01-01 00:00:00")
    periods = int(years * 365.25 * 24 / step_h)
    index = pd.date_range(start=start, periods=periods, freq=f"{interval_min}min")

    doy = index.dayofyear.to_numpy().astype(float)
    hour = index.hour.to_numpy() + index.minute.to_numpy() / 60.0
    wetness = seasonal_wetness(doy)

    # --- Synoptic regime: a slow (multi-day) wet/dry spell on top of the season.
    # Without it storm arrivals are memoryless, nothing observable now can say
    # anything about the next 48h, and the models are being asked to forecast noise.
    regime = _ou_process(rng, periods, step_h, REGIME_TIMESCALE_H)
    regime = 1.0 / (1.0 + np.exp(-1.6 * regime))  # squash to 0..1, mean ~0.5

    # Independent slow anomalies for humidity and soil drying. Real stations see
    # humid air over dry ground (dry harmattan-break days at 60% RH with a soil
    # probe still reading 4095); if humidity and soil saturation are driven by the
    # same variable alone, the model learns humidity as a proxy for wet ground and
    # then ignores the soil probe.
    humidity_anomaly = _ou_process(rng, periods, step_h, tau_h=48.0) * 9.0
    drying_anomaly = _ou_process(rng, periods, step_h, tau_h=120.0)

    # --- Storm arrivals: seasonal + regime-modulated Poisson process, each storm
    # --- preceded by a pressure fall so the sensors carry predictive information.
    rain_rate = np.zeros(periods)
    storm_forcing = np.zeros(periods)  # drives the pre-storm pressure dip
    storms_per_day = (0.08 + 1.5 * wetness) * (0.25 + 1.75 * regime)
    arrival_prob = storms_per_day * step_h / 24.0

    i = 0
    while i < periods:
        if rng.random() < arrival_prob[i]:
            duration_steps = max(1, int(rng.gamma(2.2, 1.6 / step_h)))
            peak = rng.gamma(2.0, 4.0 + 14.0 * wetness[i])
            peak = min(peak, wf.RAIN_RATE_FULL_SCALE_MM_H)
            shape = np.sin(np.linspace(0, np.pi, duration_steps + 2))[1:-1]
            end = min(periods, i + duration_steps)
            rain_rate[i:end] += peak * shape[: end - i]

            # Pressure begins falling up to 12h before the storm and recovers after.
            lead_steps = int(12 / step_h)
            lo = max(0, i - lead_steps)
            ramp = np.linspace(0.0, 1.0, i - lo) if i > lo else np.array([])
            storm_forcing[lo:i] = np.maximum(storm_forcing[lo:i], ramp * peak)
            storm_forcing[i:end] = np.maximum(storm_forcing[i:end], peak)
            i = end + int(rng.integers(1, max(2, int(6 / step_h))))
        else:
            i += 1

    rain_rate = np.clip(rain_rate, 0.0, wf.RAIN_RATE_FULL_SCALE_MM_H)

    # --- Soil bucket -----------------------------------------------------------
    storage = np.zeros(periods)
    level = 0.25 * SOIL_CAPACITY_MM
    base_drain = SOIL_DRAIN_MM_PER_DAY * step_h / 24.0
    # Drying rate varies with sun, wind and season, so identical rainfall leaves
    # the ground at different saturations - the soil probe carries information the
    # rain gauge does not.
    drain_scale = np.clip(1.0 + 0.55 * drying_anomaly + 0.5 * (1.0 - wetness), 0.3, 2.6)
    for k in range(periods):
        level += rain_rate[k] * step_h
        level -= base_drain * drain_scale[k]
        level = min(max(level, 0.0), SOIL_CAPACITY_MM)
        storage[k] = level
    saturation_pct = 100.0 * storage / SOIL_CAPACITY_MM

    # --- Pressure --------------------------------------------------------------
    synoptic = np.cumsum(rng.normal(0.0, 5.0, periods))
    synoptic -= pd.Series(synoptic).rolling(int(72 / step_h), min_periods=1).mean().to_numpy()
    semidiurnal = 110.0 * np.sin(2 * np.pi * (hour - 3) / 12.0)
    pressure_loc1 = (
        PRESSURE_BASE_PA
        + synoptic
        + semidiurnal
        - 9.0 * storm_forcing
        - 60.0 * wetness
        - 420.0 * (regime - 0.5)
        + rng.normal(0.0, 12.0, periods)
    )
    pressure_loc2 = (
        pressure_loc1
        + STATION2_PRESSURE_OFFSET_PA
        + rng.normal(0.0, 18.0, periods)
        - 2.0 * storm_forcing
    )

    # --- Temperature and humidity ---------------------------------------------
    seasonal_temp = TEMP_ANNUAL_MEAN_C + TEMP_ANNUAL_AMPLITUDE_C * np.cos(
        2 * np.pi * (doy - 60) / 365.25
    )
    diurnal_temp = TEMP_DIURNAL_AMPLITUDE_C * np.sin(2 * np.pi * (hour - 9) / 24.0)
    temperature = (
        seasonal_temp
        + diurnal_temp
        - 2.5 * wetness
        - 2.0 * (regime - 0.5)
        - 0.16 * rain_rate
        + rng.normal(0.0, 0.6, periods)
    )
    temp_bmp180 = temperature + 0.6 + rng.normal(0.0, 0.25, periods)
    temp_bmp280 = temperature + 1.1 + rng.normal(0.0, 0.25, periods)

    humidity = (
        HUMIDITY_DRY_SEASON
        + (HUMIDITY_WET_SEASON - HUMIDITY_DRY_SEASON) * wetness
        + 0.12 * saturation_pct
        + 9.0 * (regime - 0.5)
        + humidity_anomaly
        + 1.1 * rain_rate
        - 6.0 * np.sin(2 * np.pi * (hour - 9) / 24.0)
        + rng.normal(0.0, 3.0, periods)
    )
    humidity = np.clip(humidity, 20.0, 100.0)

    # --- Raw ADC readings the ESP32 would actually report ----------------------
    rain_raw = np.array([wf.rain_rate_to_adc(r) for r in rain_rate])
    rain_raw = np.clip(rain_raw + rng.normal(0.0, 25.0, periods), 0, 4095).round()
    soil_moisture = np.array([wf.soil_saturation_pct_to_adc(s) for s in saturation_pct])
    soil_moisture = np.clip(soil_moisture + rng.normal(0.0, 30.0, periods), 0, 4095).round()

    frame = pd.DataFrame(
        {
            "temperature": temperature.round(2),
            "humidity": humidity.round(2),
            "temp_bmp180": temp_bmp180.round(2),
            "pressure_loc1": pressure_loc1.round(1),
            "temp_bmp280": temp_bmp280.round(2),
            "pressure_loc2": pressure_loc2.round(1),
            "rain_raw": rain_raw,
            "soil_moisture": soil_moisture,
        },
        index=index,
    )
    frame["_rain_rate_true"] = rain_rate
    frame["_saturation_true"] = saturation_pct
    return frame


def build_targets(frame: pd.DataFrame, interval_min: int) -> pd.DataFrame:
    """48h-ahead targets, including the two continuous 0-100 index targets."""
    step_h = interval_min / 60.0
    horizon = int(FORECAST_HORIZON_H / step_h)

    out = pd.DataFrame(index=frame.index)
    out["temperature_target"] = frame["temperature"].shift(-horizon)
    out["humidity_target"] = frame["humidity"].shift(-horizon)
    out["pressure_loc1_target"] = frame["pressure_loc1"].shift(-horizon)
    out["pressure_loc2_target"] = frame["pressure_loc2"].shift(-horizon)

    # Rain that will fall over the next 48h, and how saturated the ground gets.
    rain_mm_per_step = frame["_rain_rate_true"] * step_h
    future_rain_48h = (
        rain_mm_per_step.iloc[::-1].rolling(horizon, min_periods=1).sum().iloc[::-1].shift(-1)
    )
    future_peak_sat = (
        frame["_saturation_true"].iloc[::-1].rolling(horizon, min_periods=1).max().iloc[::-1].shift(-1)
    )

    out["rain_future_48h_mm"] = future_rain_48h.round(3)
    out["peak_saturation_48h_pct"] = future_peak_sat.round(3)
    out["rain_index_target"] = [
        round(wf.rain_index_from_mm(v), 3) if pd.notna(v) else np.nan for v in future_rain_48h
    ]
    out["flood_index_target"] = [
        round(wf.flood_index_from(r, s), 3) if pd.notna(r) and pd.notna(s) else np.nan
        for r, s in zip(future_rain_48h, future_peak_sat)
    ]
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--years", type=int, default=4)
    parser.add_argument("--interval-min", type=int, default=30)
    parser.add_argument("--out", default="weather_flood_dataset.csv.gz")
    parser.add_argument("--seed", type=int, default=20250808)
    args = parser.parse_args()

    print(f"Simulating {args.years} years at {args.interval_min}-minute resolution...")
    raw = simulate(args.years, args.interval_min, args.seed)
    derived = wf.derive_features_frame(raw)
    targets = build_targets(raw, args.interval_min)

    df = pd.concat([raw[wf.RAW_SENSOR_COLUMNS], derived, targets], axis=1)

    # Calendar columns are logged for plotting and for the climatology table only.
    time_feats = pd.DataFrame(
        [wf.time_features(ts) for ts in df.index], index=df.index
    )
    df = pd.concat([time_feats, df], axis=1)
    df.insert(0, "timestamp", df.index.strftime("%Y-%m-%d %H:%M:%S"))

    # Category labels derived from the index targets, so the string labels and the
    # percentages can never disagree.
    df["flood_risk_target"] = [
        wf.flood_category(v) if pd.notna(v) else None for v in df["flood_index_target"]
    ]
    df["rain_category_target"] = [
        wf.rain_category(v) if pd.notna(v) else None for v in df["rain_index_target"]
    ]

    before = len(df)
    df = df.dropna(subset=["temperature_target", "flood_index_target", "rain_index_target"])
    df = df.reset_index(drop=True)
    print(f"Dropped {before - len(df)} tail rows without a full 48h horizon.")

    float_cols = df.select_dtypes(include="float").columns
    df[float_cols] = df[float_cols].round(3)
    df.to_csv(args.out, index=False)
    print(f"Wrote {len(df):,} rows x {df.shape[1]} columns -> {args.out}")

    print("\nLabel distribution by month (season no longer determines the label):")
    print(pd.crosstab(df["month"], df["flood_risk_target"]))
    print("\nIndex target summary:")
    print(df[["flood_index_target", "rain_index_target"]].describe().round(2))
    print("\nSensor ranges (must cover what the ESP32 actually reports):")
    print(
        df[wf.RAW_SENSOR_COLUMNS].agg(["min", "max", "mean"]).T.round(1)
    )

    print("\nSensor sanity — mean readings by flood category:")
    print(
        df.groupby("flood_risk_target")[
            ["rain_raw", "soil_moisture", "rain_accum_24h", "humidity"]
        ].mean().round(1)
    )


if __name__ == "__main__":
    main()
