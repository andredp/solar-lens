"""Feature engineering functions for Solar Lens machine learning model."""

import datetime
from typing import Any

import numpy as np


def encode_cyclical_time(hour: float | int, minute: float | int) -> tuple[float, float]:
    """Encode time of day (hour and minute) as sine and cosine components.

    This ensures that 23:59 and 00:00 are treated as close points in time.
    """
    total_minutes = hour * 60.0 + minute
    minutes_in_day = 1440.0
    angle = 2.0 * np.pi * total_minutes / minutes_in_day
    return float(np.sin(angle)), float(np.cos(angle))


def encode_cyclical_day_of_year(day_of_year: float | int) -> tuple[float, float]:
    """Encode day of year as sine and cosine components.

    This captures seasonal patterns (e.g. solar angle, temperature trends)
    cyclically, wrapping December 31st to January 1st smoothly.
    """
    days_in_year = 365.25
    angle = 2.0 * np.pi * day_of_year / days_in_year
    return float(np.sin(angle)), float(np.cos(angle))


def compute_rolling_mean(arr: np.ndarray, window: int) -> np.ndarray:
    """Calculate the rolling mean of a 1D numpy array with window size.

    Mimics pandas.Series.rolling(window=window, min_periods=1).mean() using O(N) cumsum.
    """
    cumsum = np.cumsum(arr)
    out = np.empty(len(arr), dtype=float)
    for i in range(len(arr)):
        start = max(0, i - window + 1)
        if start == 0:
            out[i] = cumsum[i] / (i + 1)
        else:
            out[i] = (cumsum[i] - cumsum[start - 1]) / window
    return out


def extract_time_features(timestamps: Any) -> dict[str, np.ndarray]:
    """Extract cyclical and categorical time features from timestamps.

    Accepts any iterable of datetime-like objects (e.g. DatetimeIndex, list of datetimes).
    Does not import or require pandas.
    """
    dt_list = list(timestamps)

    if len(dt_list) > 0 and isinstance(dt_list[0], np.datetime64):
        new_list = []
        for dt in dt_list:
            s = str(dt)
            new_list.append(
                datetime.datetime.strptime(s[:19].replace("T", " "), "%Y-%m-%d %H:%M:%S")
            )
        dt_list = new_list

    hours = np.array([dt.hour for dt in dt_list])
    minutes = np.array([dt.minute for dt in dt_list])

    dayofyears = []
    for dt in dt_list:
        if hasattr(dt, "dayofyear"):
            dayofyears.append(float(dt.dayofyear))
        else:
            dayofyears.append(float(dt.timetuple().tm_yday))

    dayofyears_arr = np.array(dayofyears)
    dayofweeks = np.array([dt.weekday() for dt in dt_list])

    total_minutes = hours * 60.0 + minutes
    minutes_in_day = 1440.0
    angle_time = 2.0 * np.pi * total_minutes / minutes_in_day
    time_sin = np.sin(angle_time)
    time_cos = np.cos(angle_time)

    days_in_year = 365.25
    angle_day = 2.0 * np.pi * dayofyears_arr / days_in_year
    day_sin = np.sin(angle_day)
    day_cos = np.cos(angle_day)

    is_weekend = np.isin(dayofweeks, [5, 6]).astype(int)

    return {
        "time_sin": time_sin,
        "time_cos": time_cos,
        "day_sin": day_sin,
        "day_cos": day_cos,
        "day_of_week": dayofweeks.astype(float),
        "is_weekend": is_weekend.astype(float),
    }
