"""Sensor platform for Solar Lens."""

from homeassistant.components.sensor import SensorEntity
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.helpers.entity_platform import AddEntitiesCallback

from .const import DOMAIN


async def async_setup_entry(
    hass: HomeAssistant, entry: ConfigEntry, async_add_entities: AddEntitiesCallback
) -> None:
    """Set up Solar Lens sensors from a config entry."""
    async_add_entities([SolarLensBatteryRemaining(entry)])


class SolarLensBatteryRemaining(SensorEntity):
    """Sensor showing estimated hours of battery remaining."""

    _attr_has_entity_name = True
    _attr_name = "Battery Hours Remaining"
    _attr_native_unit_of_measurement = "h"
    _attr_icon = "mdi:battery-clock-outline"

    def __init__(self, entry: ConfigEntry) -> None:
        """Initialize the sensor."""
        self._attr_unique_id = f"{DOMAIN}_battery_hours_remaining"
        self._entry = entry

    @property
    def native_value(self) -> float | None:
        """Return the sensor value."""
        # TODO: implement prediction logic
        return None
