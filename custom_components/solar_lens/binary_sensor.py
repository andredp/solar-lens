"""Binary sensor platform for Solar Lens."""

from __future__ import annotations

from homeassistant.components.binary_sensor import BinarySensorEntity
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.helpers.device_registry import DeviceInfo
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.helpers.update_coordinator import CoordinatorEntity

from .const import DOMAIN
from .coordinator import SolarLensCoordinator


async def async_setup_entry(
    hass: HomeAssistant, entry: ConfigEntry, async_add_entities: AddEntitiesCallback
) -> None:
    """Set up Solar Lens binary sensors from a config entry."""
    coordinator: SolarLensCoordinator = hass.data[DOMAIN][entry.entry_id]

    async_add_entities([SolarLensWillBatteryLastTheNight(coordinator)])


class SolarLensBaseBinarySensor(CoordinatorEntity[SolarLensCoordinator], BinarySensorEntity):
    """Base class for all Solar Lens binary sensors."""

    _attr_has_entity_name = True

    def __init__(self, coordinator: SolarLensCoordinator, name: str, key: str) -> None:
        """Initialize the binary sensor."""
        super().__init__(coordinator)
        self._key = key
        self._attr_name = name
        self._attr_unique_id = f"{DOMAIN}_{key}"

    @property
    def device_info(self) -> DeviceInfo | None:
        """Return device information to link entities."""
        return DeviceInfo(
            identifiers={(DOMAIN, self.coordinator.entry.entry_id)},
            name="Solar Lens",
            manufacturer="Solar Lens Team",
            model="Battery Prediction Engine",
        )


class SolarLensWillBatteryLastTheNight(SolarLensBaseBinarySensor):
    """Binary sensor showing whether the battery is predicted to last the night."""

    _attr_icon = "mdi:weather-night"

    def __init__(self, coordinator: SolarLensCoordinator) -> None:
        """Initialize the binary sensor."""
        super().__init__(coordinator, "Will Battery Last the Night", "will_battery_last_the_night")

    @property
    def is_on(self) -> bool | None:
        """Return True if the battery is predicted to last the night."""
        if not self.coordinator.data:
            return None
        val = self.coordinator.data.get("will_battery_last_the_night")
        return bool(val) if val is not None else None
