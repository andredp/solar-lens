"""Tests for SolarLensCoordinator and its sensors."""

import sys
from datetime import timedelta
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Add custom_components to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "custom_components"))

from homeassistant.util import dt as dt_util

from solar_lens.coordinator import SolarLensCoordinator
from solar_lens.sensor import (
    SolarLensBatteryEmptyTime,
    SolarLensBatteryRemaining,
    SolarLensChargeResumeTime,
    SolarLensGapHours,
    SolarLensPredictionCurve,
    SolarLensTomorrowSunsetSoCEstimate,
    SolarLensWillBatteryLastTheNight,
)


@pytest.fixture
def mock_hass() -> MagicMock:
    """Create a mock HomeAssistant instance."""
    hass = MagicMock()

    # Mock async_add_executor_job to run the function synchronously
    async def mock_async_add_job(func, *args, **kwargs):
        return func(*args, **kwargs)

    hass.async_add_executor_job = AsyncMock(side_effect=mock_async_add_job)
    return hass


@pytest.fixture
def mock_config_entry() -> MagicMock:
    """Create a mock ConfigEntry."""
    entry = MagicMock()
    entry.data = {
        "soc_entity": "sensor.battery_soc",
        "consumption_entity": "sensor.house_consumption",
        "solar_forecast_entity": "sensor.solar_forecast",
        "weather_entity": "sensor.outdoor_temperature",
        "battery_capacity_kwh": 7.2,
    }
    entry.entry_id = "test_entry_id"
    return entry


@pytest.mark.anyio
async def test_coordinator_update_success(
    mock_hass: MagicMock, mock_config_entry: MagicMock
) -> None:
    """Test successful coordinator update with mock HA state and statistics."""
    now = dt_util.utcnow()

    # 1. Mock entity states
    soc_state = MagicMock()
    soc_state.state = "50.0"  # 50% SoC

    solar_forecast_state = MagicMock()
    # Forecast.Solar style watt_hours attribute mapping ISO timestamps to Wh
    future_hour_1 = (now + timedelta(hours=1)).replace(minute=0, second=0, microsecond=0)
    future_hour_2 = (now + timedelta(hours=2)).replace(minute=0, second=0, microsecond=0)
    solar_forecast_state.attributes = {
        "watt_hours": {
            future_hour_1.isoformat(): 2000.0,  # 2000 Wh -> 2 kW
            future_hour_2.isoformat(): 1000.0,  # 1000 Wh -> 1 kW
        }
    }

    def mock_get_state(entity_id: str):
        if entity_id == "sensor.battery_soc":
            return soc_state
        if entity_id == "sensor.solar_forecast":
            return solar_forecast_state
        if entity_id == "sensor.outdoor_temperature":
            weather_state = MagicMock()
            weather_state.state = "20.0"
            weather_state.attributes = {}
            return weather_state
        return None

    mock_hass.states.get = mock_get_state

    # 2. Mock historical statistics for consumption & temperature
    # Generate 48 hours of statistics (2 days of history)
    mock_stats = []
    mock_temp_stats = []
    for i in range(48):
        stat_time = now - timedelta(hours=48 - i)
        mock_stats.append(
            {
                "start": stat_time.timestamp(),
                "mean": 0.4 if i % 24 < 8 or i % 24 > 22 else 1.2,  # daily consumption pattern
            }
        )
        mock_temp_stats.append(
            {
                "start": stat_time.timestamp(),
                "mean": 18.0,
            }
        )

    mock_stats_dict = {
        "sensor.house_consumption": mock_stats,
        "sensor.outdoor_temperature": mock_temp_stats,
    }

    # Patch statistics_during_period to return our mock stats
    with patch(
        "solar_lens.coordinator.statistics_during_period",
        return_value=mock_stats_dict,
    ):
        coordinator = SolarLensCoordinator(mock_hass, mock_config_entry)

        # Force a refresh
        await coordinator.async_refresh()

        assert coordinator.last_update_success
        data = coordinator.data

        # Verify output keys and types
        assert "battery_hours_remaining" in data
        assert "battery_empty_time" in data
        assert "charge_resume_time" in data
        assert "gap_hours" in data
        assert "tomorrow_sunset_soc_estimate" in data
        assert "will_battery_last_the_night" in data
        assert "prediction_curve" in data

        # Check prediction curve length (288 steps/day * 2 days = 576 steps)
        assert len(data["prediction_curve"]) == 576

        # Test the sensors using the coordinator data
        sensor_remaining = SolarLensBatteryRemaining(coordinator)
        sensor_empty = SolarLensBatteryEmptyTime(coordinator)
        sensor_resume = SolarLensChargeResumeTime(coordinator)
        sensor_gap = SolarLensGapHours(coordinator)
        sensor_sunset = SolarLensTomorrowSunsetSoCEstimate(coordinator)
        sensor_will_last = SolarLensWillBatteryLastTheNight(coordinator)
        sensor_curve = SolarLensPredictionCurve(coordinator)

        assert sensor_remaining.native_value == data["battery_hours_remaining"]
        assert sensor_empty.native_value == data["battery_empty_time"]
        assert sensor_resume.native_value == data["charge_resume_time"]
        assert sensor_gap.native_value == data["gap_hours"]
        assert sensor_sunset.native_value == data["tomorrow_sunset_soc_estimate"]
        assert sensor_will_last.native_value == data["will_battery_last_the_night"]
        assert sensor_curve.native_value == len(data["prediction_curve"])
        assert sensor_curve.extra_state_attributes["prediction_curve"] == data["prediction_curve"]
