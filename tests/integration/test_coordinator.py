"""Tests for SolarLensCoordinator and its sensors."""

import sys
from datetime import timedelta
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Add custom_components to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "custom_components"))

from homeassistant.util import dt as dt_util

from solar_lens.binary_sensor import SolarLensWillBatteryLastTheNight
from solar_lens.coordinator import SolarLensCoordinator
from solar_lens.sensor import (
    SolarLensBatteryEmptyTime,
    SolarLensBatteryRemaining,
    SolarLensChargeResumeTime,
    SolarLensGapHours,
    SolarLensPredictionCurve,
    SolarLensTomorrowSunsetSoCEstimate,
)


@pytest.fixture
def mock_hass() -> MagicMock:
    """Create a mock HomeAssistant instance."""
    hass = MagicMock()

    # Mock async_add_executor_job to run the function synchronously
    async def mock_async_add_job(func: Any, *args: Any, **kwargs: Any) -> Any:
        return func(*args, **kwargs)

    hass.async_add_executor_job = AsyncMock(side_effect=mock_async_add_job)

    # Mock weather.get_forecasts service call to verify parsing
    async def mock_async_call(
        domain: str, service: str, service_data: dict[str, Any], **kwargs: Any
    ) -> dict[str, Any]:
        if domain == "weather" and service == "get_forecasts":
            # Return hourly temperature forecast
            entity_id = service_data.get("entity_id")
            if not isinstance(entity_id, str):
                entity_id = "weather.home"
            return {
                entity_id: {
                    "forecast": [
                        {
                            "datetime": (dt_util.utcnow() + timedelta(hours=i)).isoformat(),
                            "temperature": 20.0,
                        }
                        for i in range(49)
                    ]
                }
            }
        return {}

    hass.services.async_call = AsyncMock(side_effect=mock_async_call)

    # Mock async_create_background_task
    def mock_async_create_task(target: Any, name: str | None = None) -> Any:
        task = MagicMock()
        task.done.return_value = True
        return task

    hass.async_create_background_task = MagicMock(side_effect=mock_async_create_task)
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
        "actual_solar_entity": "sensor.actual_solar",
        "charge_limit_entity": "sensor.charge_limit",
        "discharge_limit_entity": "sensor.discharge_limit",
        "battery_voltage_entity": "sensor.battery_voltage",
        "battery_temp_entity": "sensor.battery_temp",
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
    future_hour_1 = (now + timedelta(hours=1)).replace(minute=0, second=0, microsecond=0)
    future_hour_2 = (now + timedelta(hours=2)).replace(minute=0, second=0, microsecond=0)
    solar_forecast_state.attributes = {
        "watt_hours": {
            future_hour_1.isoformat(): 2000.0,  # 2000 Wh -> 2 kW
            future_hour_2.isoformat(): 1000.0,  # 1000 Wh -> 1 kW
        }
    }

    def mock_get_state(entity_id: str) -> Any:
        if entity_id == "sensor.battery_soc":
            return soc_state
        if entity_id == "sensor.solar_forecast":
            return solar_forecast_state
        if entity_id == "sensor.outdoor_temperature":
            weather_state = MagicMock()
            weather_state.state = "20.0"
            weather_state.attributes = {}
            return weather_state
        if entity_id == "sensor.actual_solar":
            actual_solar_state = MagicMock()
            actual_solar_state.state = "1.5"
            return actual_solar_state
        if entity_id == "sensor.charge_limit":
            charge_limit_state = MagicMock()
            charge_limit_state.state = "75.0"
            return charge_limit_state
        if entity_id == "sensor.discharge_limit":
            discharge_limit_state = MagicMock()
            discharge_limit_state.state = "75.0"
            return discharge_limit_state
        if entity_id == "sensor.battery_voltage":
            voltage_state = MagicMock()
            voltage_state.state = "50.0"
            return voltage_state
        if entity_id == "sensor.battery_temp":
            temp_state = MagicMock()
            temp_state.state = "22.0"
            temp_state.attributes = {"unit_of_measurement": "°C"}
            return temp_state
        return None

    mock_hass.states.get = mock_get_state

    # 2. Mock historical statistics for all entities
    # Generate 48 hours of statistics (2 days of history)
    mock_stats = []
    mock_temp_stats = []
    mock_actual_solar_stats = []
    mock_charge_limit_stats = []
    mock_discharge_limit_stats = []
    mock_battery_temp_stats = []
    mock_battery_voltage_stats = []

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
        mock_actual_solar_stats.append(
            {
                "start": stat_time.timestamp(),
                "mean": 1.0 if 8 <= i % 24 <= 17 else 0.0,
            }
        )
        mock_charge_limit_stats.append(
            {
                "start": stat_time.timestamp(),
                "mean": 75.0,
            }
        )
        mock_discharge_limit_stats.append(
            {
                "start": stat_time.timestamp(),
                "mean": 75.0,
            }
        )
        mock_battery_temp_stats.append(
            {
                "start": stat_time.timestamp(),
                "mean": 22.0,
            }
        )
        mock_battery_voltage_stats.append(
            {
                "start": stat_time.timestamp(),
                "mean": 50.0,
            }
        )

    mock_stats_dict = {
        "sensor.house_consumption": mock_stats,
        "sensor.outdoor_temperature": mock_temp_stats,
        "sensor.actual_solar": mock_actual_solar_stats,
        "sensor.charge_limit": mock_charge_limit_stats,
        "sensor.discharge_limit": mock_discharge_limit_stats,
        "sensor.battery_temp": mock_battery_temp_stats,
        "sensor.battery_voltage": mock_battery_voltage_stats,
        "sensor.solar_forecast": mock_actual_solar_stats,
    }

    # Patch statistics_during_period to return our mock stats
    with patch(
        "solar_lens.coordinator.statistics_during_period",
        return_value=mock_stats_dict,
    ):
        coordinator = SolarLensCoordinator(mock_hass, mock_config_entry)

        # Retrain models first to make sure they fit
        await coordinator._async_retrain_models()

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
        assert sensor_will_last.is_on == data["will_battery_last_the_night"]
        assert sensor_curve.native_value == len(data["prediction_curve"])
        extra_attrs = sensor_curve.extra_state_attributes
        assert extra_attrs is not None
        assert extra_attrs["prediction_curve"] == data["prediction_curve"]


@pytest.mark.anyio
async def test_coordinator_update_success_missing_optional_entities(
    mock_hass: MagicMock,
) -> None:
    """Test successful coordinator update when optional entities are not configured."""
    now = dt_util.utcnow()

    # Config entry with only required entities
    entry = MagicMock()
    entry.data = {
        "soc_entity": "sensor.battery_soc",
        "consumption_entity": "sensor.house_consumption",
        "solar_forecast_entity": "sensor.solar_forecast",
        "weather_entity": "sensor.outdoor_temperature",
        "battery_capacity_kwh": 7.2,
    }
    entry.entry_id = "test_entry_id"

    # 1. Mock entity states
    soc_state = MagicMock()
    soc_state.state = "50.0"

    solar_forecast_state = MagicMock()
    future_hour_1 = (now + timedelta(hours=1)).replace(minute=0, second=0, microsecond=0)
    solar_forecast_state.attributes = {
        "watt_hours": {
            future_hour_1.isoformat(): 2000.0,
        }
    }

    def mock_get_state(entity_id: str) -> Any:
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

    # 2. Mock stats for required entities only
    mock_stats = []
    mock_temp_stats = []
    for i in range(48):
        stat_time = now - timedelta(hours=48 - i)
        mock_stats.append(
            {
                "start": stat_time.timestamp(),
                "mean": 0.5,
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
        "sensor.solar_forecast": mock_stats,
    }

    with patch(
        "solar_lens.coordinator.statistics_during_period",
        return_value=mock_stats_dict,
    ):
        coordinator = SolarLensCoordinator(mock_hass, entry)

        # Retrain models first to make sure they fit fallback paths
        await coordinator._async_retrain_models()

        # Force a refresh
        await coordinator.async_refresh()

        assert coordinator.last_update_success
        data = coordinator.data

        # Verify output structure and fallback values
        assert "battery_hours_remaining" in data
        assert len(data["prediction_curve"]) == 576


def test_sensors_with_empty_data() -> None:
    """Verify sensors return None when coordinator has no data."""
    coordinator = MagicMock()
    coordinator.data = None

    sensor_remaining = SolarLensBatteryRemaining(coordinator)
    sensor_empty = SolarLensBatteryEmptyTime(coordinator)
    sensor_resume = SolarLensChargeResumeTime(coordinator)
    sensor_gap = SolarLensGapHours(coordinator)
    sensor_sunset = SolarLensTomorrowSunsetSoCEstimate(coordinator)
    sensor_curve = SolarLensPredictionCurve(coordinator)
    sensor_will_last = SolarLensWillBatteryLastTheNight(coordinator)

    assert sensor_remaining.native_value is None
    assert sensor_empty.native_value is None
    assert sensor_resume.native_value is None
    assert sensor_gap.native_value is None
    assert sensor_sunset.native_value is None
    assert sensor_curve.native_value is None
    assert sensor_curve.extra_state_attributes is None
    assert sensor_will_last.is_on is None
