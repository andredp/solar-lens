"""Tests for the Solar Lens config flow."""

import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

# Add custom_components to path so we can import solar_lens
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "custom_components"))

from homeassistant.data_entry_flow import FlowResultType

from solar_lens.config_flow import SolarLensConfigFlow
from solar_lens.const import DOMAIN


@pytest.mark.anyio
async def test_config_flow_user_step() -> None:
    """Test config flow user step (both form and creation)."""
    flow = SolarLensConfigFlow()
    flow.hass = MagicMock()
    flow.context = {}

    # Mock parent class methods called during entry creation
    flow.async_set_unique_id = AsyncMock()
    flow._abort_if_unique_id_configured = MagicMock()
    flow.async_create_entry = MagicMock(
        side_effect=lambda title, data: {
            "type": FlowResultType.CREATE_ENTRY,
            "title": title,
            "data": data,
        }
    )

    # 1. Test user step without input (should show form)
    # We temporarily remove the mock for user_input = None to allow the schema to show
    result = await flow.async_step_user()
    assert result["type"] == FlowResultType.FORM
    assert result["step_id"] == "user"

    # 2. Test user step with input (should create entry)
    user_input = {
        "soc_entity": "sensor.battery_soc",
        "consumption_entity": "sensor.house_consumption",
        "solar_forecast_entity": "sensor.solar_forecast",
        "weather_entity": "sensor.outdoor_temperature",
        "battery_capacity_kwh": 7.2,
    }

    result2 = await flow.async_step_user(user_input)

    flow.async_set_unique_id.assert_called_once_with(DOMAIN)
    flow._abort_if_unique_id_configured.assert_called_once()
    assert result2["type"] == FlowResultType.CREATE_ENTRY
    assert result2["title"] == "Solar Lens"
    assert result2["data"] == user_input
