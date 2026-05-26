"""Basic tests for Solar Lens integration."""

import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Add custom_components to path so we can import solar_lens
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "custom_components"))

from solar_lens import async_setup_entry, async_unload_entry
from solar_lens.const import DOMAIN


def test_domain_constant() -> None:
    """Verify that DOMAIN constant is correct."""
    assert DOMAIN == "solar_lens"


@pytest.mark.anyio
async def test_async_setup_unload_entry() -> None:
    """Test setting up and unloading a config entry."""
    hass = MagicMock()
    hass.data = {}

    # Mock config entry
    entry = MagicMock()
    entry.entry_id = "test_entry"
    entry.data = {
        "soc_entity": "sensor.battery_soc",
        "consumption_entity": "sensor.house_consumption",
        "solar_forecast_entity": "sensor.solar_forecast",
        "weather_entity": "sensor.outdoor_temperature",
        "battery_capacity_kwh": 7.2,
    }

    # Mock Coordinator
    mock_coordinator = MagicMock()
    mock_coordinator.async_config_entry_first_refresh = AsyncMock()

    # Mock async_forward_entry_setups
    hass.config_entries.async_forward_entry_setups = AsyncMock(return_value=True)
    hass.config_entries.async_unload_platforms = AsyncMock(return_value=True)

    with patch("solar_lens.SolarLensCoordinator", return_value=mock_coordinator) as mock_coord_init:
        # Call setup
        assert await async_setup_entry(hass, entry) is True

        # Assertions
        mock_coord_init.assert_called_once_with(hass, entry)
        mock_coordinator.async_config_entry_first_refresh.assert_called_once()
        hass.config_entries.async_forward_entry_setups.assert_called_once_with(
            entry, ["sensor", "binary_sensor"]
        )
        assert hass.data[DOMAIN][entry.entry_id] == mock_coordinator

    # Test unload
    assert await async_unload_entry(hass, entry) is True
    hass.config_entries.async_unload_platforms.assert_called_once_with(
        entry, ["sensor", "binary_sensor"]
    )
    assert entry.entry_id not in hass.data[DOMAIN]
