"""
Single source of truth for sensor calibration, feature engineering, index scales
and the climatology blend used by this project.

Every consumer imports from here — the dataset generator, the training script and
the Flask API — so a feature can never be defined one way at training time and a
different way at inference time.
"""

from __future__ import annotations

import bisect
import math
from collections.abc import Sequence

import numpy as np
import pandas as pd

# ------------------------------------------------------------------------------
# Sensor calibration
# ------------------------------------------------------------------------------
# Both analog sensors are "dry is high": a bare, dry plate/probe reads close to
# the ESP32's 12-bit full scale (4095) and the reading falls as water bridges the
# electrodes. Recalibrate these four constants against your own hardware by
# reading the raw ADC value with the sensor dry, then fully wet.
RAIN_ADC_DRY = 4095.0      # rain plate completely dry
RAIN_ADC_WET = 1200.0      # rain plate streaming with water
SOIL_ADC_DRY = 4095.0      # probe in dry air / bone-dry soil
SOIL_ADC_SAT = 1000.0      # probe in fully saturated soil

# Rain rate (mm/h) assigned to a fully wet plate. A tipping-bucket gauge would
# give a true rate; a resistive plate only gives wetness, so this maps the
# wetness fraction onto a physical rate.
RAIN_RATE_FULL_SCALE_MM_H = 40.0

# ------------------------------------------------------------------------------
# Index scales (all indices are 0-100)
# ------------------------------------------------------------------------------
# Rainfall index reaches 100 at this much accumulation over the next 48h.
RAIN_INDEX_FULL_SCALE_MM = 120.0
# Flood index weighting: how much of the risk comes from incoming rainfall vs
# how saturated the ground already is (antecedent wetness).
FLOOD_RAIN_WEIGHT = 0.60
FLOOD_SATURATION_WEIGHT = 0.40

# Temperature percent scale: 0% at TEMP_PERCENT_MIN_C, 100% at TEMP_PERCENT_MAX_C.
TEMP_PERCENT_MIN_C = 15.0
TEMP_PERCENT_MAX_C = 45.0

# With no stored history, credit the current rain rate with this many hours of
# accumulation rather than extrapolating it across the full 6h/24h window.
COLD_START_PERSISTENCE_H = 1.0

# ------------------------------------------------------------------------------
# Blend weights — the date/season half and the sensor half of every index
# ------------------------------------------------------------------------------
CLIMATE_WEIGHT = 0.50
SENSOR_WEIGHT = 0.50

# ------------------------------------------------------------------------------
# Category thresholds (kept so existing clients that read the string labels,
# including the ESP32 firmware and the dashboard, keep working)
# ------------------------------------------------------------------------------
FLOOD_CATEGORY_THRESHOLDS = [(25.0, "low"), (60.0, "watch")]
FLOOD_CATEGORY_TOP = "high"
RAIN_CATEGORY_THRESHOLDS = [(20.0, "dry"), (45.0, "light"), (70.0, "moderate")]
RAIN_CATEGORY_TOP = "heavy"

# ------------------------------------------------------------------------------
# Column groups
# ------------------------------------------------------------------------------
RAW_SENSOR_COLUMNS = [
    "temperature",
    "humidity",
    "temp_bmp180",
    "pressure_loc1",
    "temp_bmp280",
    "pressure_loc2",
    "rain_raw",
    "soil_moisture",
]

# Derived from the raw readings plus recent history. Note every one of these is
# a *sensor* quantity — no calendar information leaks in.
DERIVED_SENSOR_COLUMNS = [
    "soil_saturation_pct",
    "rain_rate",
    "rain_accum_6h",
    "rain_accum_24h",
    "pressure_loc1_trend_3h",
    "pressure_loc1_trend_6h",
    "pressure_loc2_trend_3h",
    "pressure_loc2_trend_6h",
    "pressure_diff",
    "humidity_trend_3h",
    "temperature_trend_3h",
]

# The model input vector. Deliberately excludes hour_*/doy_*/month: the calendar
# enters the prediction only through the climatology term, at a fixed weight, so
# it can never outvote the sensors (see README / manifest "blend_weights").
SENSOR_FEATURE_COLUMNS = RAW_SENSOR_COLUMNS + DERIVED_SENSOR_COLUMNS

# Logged for plotting and for building the climatology table — never fed to a model.
TIME_COLUMNS = ["hour_sin", "hour_cos", "doy_sin", "doy_cos", "month"]

REGRESSION_TARGETS = [
    "temperature_target",
    "humidity_target",
    "pressure_loc1_target_delta",
    "pressure_loc2_target_delta",
]
INDEX_TARGETS = ["flood_index_target", "rain_index_target"]

PRESSURE_PAIRS = [
    ("pressure_loc1", "pressure_loc1_target", "pressure_loc1_target_delta"),
    ("pressure_loc2", "pressure_loc2_target", "pressure_loc2_target_delta"),
]


# ------------------------------------------------------------------------------
# Scalar conversions
# ------------------------------------------------------------------------------
def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def rain_wetness_fraction(adc: float) -> float:
    """0.0 = plate dry, 1.0 = plate saturated."""
    span = RAIN_ADC_DRY - RAIN_ADC_WET
    return clamp((RAIN_ADC_DRY - float(adc)) / span, 0.0, 1.0)


def rain_adc_to_rate_mm_h(adc: float) -> float:
    """Instantaneous rain rate implied by a raw rain-plate ADC reading."""
    return rain_wetness_fraction(adc) * RAIN_RATE_FULL_SCALE_MM_H


def rain_rate_to_adc(rate_mm_h: float) -> float:
    """Inverse of rain_adc_to_rate_mm_h, used by the dataset generator."""
    wetness = clamp(rate_mm_h / RAIN_RATE_FULL_SCALE_MM_H, 0.0, 1.0)
    return RAIN_ADC_DRY - wetness * (RAIN_ADC_DRY - RAIN_ADC_WET)


def soil_adc_to_saturation_pct(adc: float) -> float:
    """0% = bone dry, 100% = saturated."""
    span = SOIL_ADC_DRY - SOIL_ADC_SAT
    return clamp((SOIL_ADC_DRY - float(adc)) / span, 0.0, 1.0) * 100.0


def soil_saturation_pct_to_adc(saturation_pct: float) -> float:
    """Inverse of soil_adc_to_saturation_pct, used by the dataset generator."""
    frac = clamp(saturation_pct / 100.0, 0.0, 1.0)
    return SOIL_ADC_DRY - frac * (SOIL_ADC_DRY - SOIL_ADC_SAT)


def rain_index_from_mm(accum_48h_mm: float) -> float:
    """0-100 rainfall tendency from a 48h accumulation in mm."""
    return clamp(accum_48h_mm / RAIN_INDEX_FULL_SCALE_MM, 0.0, 1.0) * 100.0


def flood_index_from(accum_48h_mm: float, peak_saturation_pct: float) -> float:
    """
    0-100 flood tendency: incoming rainfall combined with how much room the
    ground has left to absorb it.
    """
    rain_part = clamp(accum_48h_mm / RAIN_INDEX_FULL_SCALE_MM, 0.0, 1.0) * 100.0
    sat_part = clamp(peak_saturation_pct, 0.0, 100.0)
    return FLOOD_RAIN_WEIGHT * rain_part + FLOOD_SATURATION_WEIGHT * sat_part


def temperature_to_percent(temp_c: float) -> float:
    """Position of a temperature on the 0-100 cold->heat scale."""
    span = TEMP_PERCENT_MAX_C - TEMP_PERCENT_MIN_C
    return clamp((float(temp_c) - TEMP_PERCENT_MIN_C) / span, 0.0, 1.0) * 100.0


def flood_category(percent: float) -> str:
    for limit, label in FLOOD_CATEGORY_THRESHOLDS:
        if percent < limit:
            return label
    return FLOOD_CATEGORY_TOP


def rain_category(percent: float) -> str:
    for limit, label in RAIN_CATEGORY_THRESHOLDS:
        if percent < limit:
            return label
    return RAIN_CATEGORY_TOP


def blend_percent(climate_percent: float, sensor_percent: float) -> float:
    """The 50/50 season + sensor combination that produces every final index."""
    return CLIMATE_WEIGHT * float(climate_percent) + SENSOR_WEIGHT * float(sensor_percent)


# ------------------------------------------------------------------------------
# Time features (logged only — see TIME_COLUMNS)
# ------------------------------------------------------------------------------
def time_features(dt) -> dict:
    hour = dt.hour + dt.minute / 60.0
    doy = dt.timetuple().tm_yday
    return {
        "hour_sin": math.sin(2 * math.pi * hour / 24.0),
        "hour_cos": math.cos(2 * math.pi * hour / 24.0),
        "doy_sin": math.sin(2 * math.pi * doy / 365.25),
        "doy_cos": math.cos(2 * math.pi * doy / 365.25),
        "month": dt.month,
    }


# ------------------------------------------------------------------------------
# Feature engineering over a history window
# ------------------------------------------------------------------------------
def _value_at_or_before(times: np.ndarray, values: np.ndarray, cutoff: float):
    """Most recent sample at or before `cutoff` (both as epoch seconds)."""
    idx = bisect.bisect_right(times, cutoff) - 1
    if idx < 0:
        return None
    return float(values[idx])


def _integrate_mm(times: np.ndarray, rates: np.ndarray, start: float, end: float) -> float:
    """
    Trapezoidal integral of a mm/h rate series over [start, end] in epoch
    seconds, returning millimetres.

    This is what makes accumulation independent of the posting interval: the ESP32
    posting every 30s and the dataset generator stepping every 30min produce the
    same accumulation for the same weather, whereas summing raw samples would not.
    """
    if len(times) == 0:
        return 0.0
    mask = (times >= start) & (times <= end)
    t = times[mask]
    r = rates[mask]
    if len(t) == 0:
        # No sample inside the window: hold the last known rate across it.
        held = _value_at_or_before(times, rates, end)
        return 0.0 if held is None else held * (end - start) / 3600.0
    if t[0] > start:
        prior = _value_at_or_before(times, rates, start)
        if prior is not None:
            t = np.concatenate(([start], t))
            r = np.concatenate(([prior], r))
    if t[-1] < end:
        t = np.concatenate((t, [end]))
        r = np.concatenate((r, [r[-1]]))
    if len(t) < 2:
        return 0.0
    return float(np.trapz(r, t) / 3600.0)


def derive_features(
    current: dict,
    now,
    history: pd.DataFrame | None = None,
) -> dict:
    """
    Build every DERIVED_SENSOR_COLUMNS value for one reading.

    `current` holds the raw sensor fields (RAW_SENSOR_COLUMNS). `history` is an
    optional DataFrame of previous readings with a `timestamp` column plus the raw
    sensor columns, ordered oldest-first. Missing history degrades to zero trends
    and to accumulation from the current rate alone, never to a fabricated value.

    On a cold start the current rate is credited for COLD_START_PERSISTENCE_H only.
    Assuming instead that the present rate had held for the whole 6h/24h window
    would report ~900 mm during a downpour, far past anything in the training
    range, and every accumulation feature would arrive at the model clipped.
    """
    rain_rate_now = rain_adc_to_rate_mm_h(current["rain_raw"])
    now_epoch = pd.Timestamp(now).timestamp()

    features = {
        "soil_saturation_pct": soil_adc_to_saturation_pct(current["soil_moisture"]),
        "rain_rate": rain_rate_now,
        "pressure_diff": float(current["pressure_loc1"]) - float(current["pressure_loc2"]),
        "pressure_loc1_trend_3h": 0.0,
        "pressure_loc1_trend_6h": 0.0,
        "pressure_loc2_trend_3h": 0.0,
        "pressure_loc2_trend_6h": 0.0,
        "humidity_trend_3h": 0.0,
        "temperature_trend_3h": 0.0,
        "rain_accum_6h": rain_rate_now * COLD_START_PERSISTENCE_H,
        "rain_accum_24h": rain_rate_now * COLD_START_PERSISTENCE_H,
    }

    if history is None or history.empty:
        return features

    hist = history.dropna(subset=["timestamp"]).copy()
    hist["_epoch"] = pd.to_datetime(hist["timestamp"]).astype("int64") / 1e9
    hist = hist.sort_values("_epoch")
    times = hist["_epoch"].to_numpy()

    rates = np.array([rain_adc_to_rate_mm_h(v) for v in hist["rain_raw"].to_numpy()])
    times_with_now = np.concatenate((times, [now_epoch]))
    rates_with_now = np.concatenate((rates, [rain_rate_now]))

    features["rain_accum_6h"] = _integrate_mm(
        times_with_now, rates_with_now, now_epoch - 6 * 3600, now_epoch
    )
    features["rain_accum_24h"] = _integrate_mm(
        times_with_now, rates_with_now, now_epoch - 24 * 3600, now_epoch
    )

    for column, hours, name in [
        ("pressure_loc1", 3, "pressure_loc1_trend_3h"),
        ("pressure_loc1", 6, "pressure_loc1_trend_6h"),
        ("pressure_loc2", 3, "pressure_loc2_trend_3h"),
        ("pressure_loc2", 6, "pressure_loc2_trend_6h"),
        ("humidity", 3, "humidity_trend_3h"),
        ("temperature", 3, "temperature_trend_3h"),
    ]:
        past = _value_at_or_before(times, hist[column].to_numpy(), now_epoch - hours * 3600)
        if past is not None:
            features[name] = float(current[column]) - past

    return features


def derive_features_frame(df: pd.DataFrame) -> pd.DataFrame:
    """
    Vectorised equivalent of derive_features for a complete, regularly sampled
    time series (used by the dataset generator). `df` must have a DatetimeIndex
    and the raw sensor columns.
    """
    step = pd.Timedelta(df.index[1] - df.index[0])
    step_h = step.total_seconds() / 3600.0

    out = pd.DataFrame(index=df.index)
    out["soil_saturation_pct"] = df["soil_moisture"].map(soil_adc_to_saturation_pct)
    out["rain_rate"] = df["rain_raw"].map(rain_adc_to_rate_mm_h)

    # Trapezoidal accumulation on a regular grid reduces to a centred rolling sum.
    mm_per_step = out["rain_rate"] * step_h
    for hours in (6, 24):
        window = max(1, int(round(hours / step_h)))
        out[f"rain_accum_{hours}h"] = (
            mm_per_step.rolling(window, min_periods=1).sum().astype(float)
        )

    for column, hours, name in [
        ("pressure_loc1", 3, "pressure_loc1_trend_3h"),
        ("pressure_loc1", 6, "pressure_loc1_trend_6h"),
        ("pressure_loc2", 3, "pressure_loc2_trend_3h"),
        ("pressure_loc2", 6, "pressure_loc2_trend_6h"),
        ("humidity", 3, "humidity_trend_3h"),
        ("temperature", 3, "temperature_trend_3h"),
    ]:
        lag = max(1, int(round(hours / step_h)))
        out[name] = (df[column] - df[column].shift(lag)).fillna(0.0)

    out["pressure_diff"] = df["pressure_loc1"] - df["pressure_loc2"]
    return out[DERIVED_SENSOR_COLUMNS]


# ------------------------------------------------------------------------------
# Climatology (the "date lookup" half of every index)
# ------------------------------------------------------------------------------
def build_climatology(
    timestamps: Sequence, values_by_name: dict, smooth_days: int = 15
) -> dict:
    """
    Average each named series by day-of-year and smooth it with a circular
    rolling mean, producing the seasonal baseline used at inference time.

    Returns {"days": [1..366], "<name>": [...366 values...], ...}.
    """
    doy = pd.DatetimeIndex(timestamps).dayofyear
    days = np.arange(1, 367)
    table = {"days": days.tolist(), "smooth_days": smooth_days}

    for name, series in values_by_name.items():
        by_doy = pd.Series(np.asarray(series, dtype=float)).groupby(doy).mean()
        full = by_doy.reindex(days)
        full = full.interpolate(limit_direction="both")
        # Circular smoothing so 31 Dec and 1 Jan stay continuous.
        tripled = pd.concat([full, full, full], ignore_index=True)
        smoothed = tripled.rolling(smooth_days, center=True, min_periods=1).mean()
        table[name] = smoothed.iloc[len(full) : 2 * len(full)].round(4).tolist()

    return table


def climatology_lookup(table: dict, name: str, dt) -> float:
    """Linear interpolation of a climatology series at a given datetime."""
    values = table[name]
    doy = dt.timetuple().tm_yday
    hour_frac = (dt.hour + dt.minute / 60.0) / 24.0
    lo = (doy - 1) % len(values)
    hi = doy % len(values)
    return float(values[lo] * (1.0 - hour_frac) + values[hi] * hour_frac)


def clip_to_training_range(features: dict, ranges: dict) -> tuple[dict, list]:
    """
    Clip each feature into the min/max seen during training.

    Gradient-boosted trees have no notion of "impossible input": a feature that
    lands far outside the training range silently pins every split in one
    direction, which is exactly how a scale mistake turns into a confident wrong
    answer. Clipping makes that degrade gracefully, and the returned list of
    clipped feature names is surfaced in the API response so it is visible.
    """
    clipped: list = []
    result = dict(features)
    for name, bounds in ranges.items():
        if name not in result:
            continue
        value = float(result[name])
        low, high = float(bounds["min"]), float(bounds["max"])
        if value < low or value > high:
            result[name] = clamp(value, low, high)
            clipped.append(name)
    return result, clipped
