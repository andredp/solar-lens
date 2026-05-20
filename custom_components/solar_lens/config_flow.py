"""Config flow for Solar Lens."""

from typing import Any

import voluptuous as vol
from homeassistant.config_entries import ConfigFlow, ConfigFlowResult

from .const import DOMAIN

STEP_USER_DATA_SCHEMA = vol.Schema(
    {
        vol.Required("soc_entity"): str,
        vol.Required("consumption_entity"): str,
        vol.Required("solar_forecast_entity"): str,
        vol.Required("battery_capacity_kwh"): vol.Coerce(float),
    }
)


class SolarLensConfigFlow(ConfigFlow, domain=DOMAIN):
    """Handle a config flow for Solar Lens."""

    VERSION = 1

    async def async_step_user(self, user_input: dict[str, Any] | None = None) -> ConfigFlowResult:
        """Handle the initial step."""
        if user_input is not None:
            await self.async_set_unique_id(DOMAIN)
            self._abort_if_unique_id_configured()
            return self.async_create_entry(title="Solar Lens", data=user_input)

        return self.async_show_form(step_id="user", data_schema=STEP_USER_DATA_SCHEMA)
