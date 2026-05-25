"""Unit tests for ML feature engineering logic."""

import sys
from pathlib import Path

import numpy as np
import pytest

# Add custom_components to path so we can import solar_lens
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "custom_components"))

from solar_lens.ml.feature_engineering import (
    compute_rolling_mean,
    encode_cyclical_day_of_year,
    encode_cyclical_time,
    extract_time_features,
)


def test_encode_cyclical_time() -> None:
    """Test that time encoding creates valid coordinates on the unit circle."""
    # Test midnight (00:00)
    sin_mid, cos_mid = encode_cyclical_time(0, 0)
    assert pytest.approx(sin_mid) == 0.0
    assert pytest.approx(cos_mid) == 1.0

    # Test noon (12:00)
    sin_noon, cos_noon = encode_cyclical_time(12, 0)
    assert pytest.approx(sin_noon) == 0.0
    assert pytest.approx(cos_noon) == -1.0

    # Test unit circle property: sin^2 + cos^2 = 1
    for h in range(24):
        for m in range(0, 60, 15):
            s, c = encode_cyclical_time(h, m)
            assert pytest.approx(s**2 + c**2) == 1.0

    # Test distance between 23:59 and 00:01 is close
    s1, c1 = encode_cyclical_time(23, 59)
    s2, c2 = encode_cyclical_time(0, 1)
    dist = np.sqrt((s1 - s2) ** 2 + (c1 - c2) ** 2)
    # Total minutes in day = 1440. Angle diff is 2 mins -> 2/1440 * 2pi = 0.0087 rad.
    # Linear distance is very small
    assert dist < 0.01


def test_encode_cyclical_day_of_year() -> None:
    """Test day of year encoding."""
    # Test start of year
    sin_start, cos_start = encode_cyclical_day_of_year(0)
    assert pytest.approx(sin_start) == 0.0
    assert pytest.approx(cos_start) == 1.0

    # Test mid year (approx day 182.6)
    s, c = encode_cyclical_day_of_year(182.6)
    assert pytest.approx(s, abs=1e-2) == 0.0
    assert pytest.approx(c, abs=1e-2) == -1.0


def test_extract_time_features() -> None:
    """Test conversion of datetime inputs to features."""
    import datetime

    timestamps = [
        datetime.datetime(2026, 6, 20, 0, 0),
        datetime.datetime(2026, 6, 20, 6, 0),
        datetime.datetime(2026, 6, 20, 12, 0),
        datetime.datetime(2026, 6, 20, 18, 0),
        datetime.datetime(2026, 6, 21, 0, 0),
    ]

    features = extract_time_features(timestamps)

    assert "time_sin" in features
    assert "time_cos" in features
    assert "day_sin" in features
    assert "day_cos" in features
    assert "day_of_week" in features
    assert "is_weekend" in features

    # 2026-06-20 is a Saturday
    assert features["is_weekend"][0] == 1
    # 2026-06-21 is a Sunday
    assert features["is_weekend"][4] == 1


def test_compute_rolling_mean() -> None:
    """Test rolling window features calculation."""
    arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    out = compute_rolling_mean(arr, 2)

    # Check values
    # row 0: mean of [1.0] = 1.0
    assert out[0] == 1.0
    # row 1: mean of [1.0, 2.0] = 1.5
    assert out[1] == 1.5
    # row 2: mean of [2.0, 3.0] = 2.5
    assert out[2] == 2.5
