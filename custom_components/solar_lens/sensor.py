"""Sensor platform for Solar Lens."""

from __future__ import annotations

from typing import Any

from homeassistant.components.sensor import SensorDeviceClass, SensorEntity
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import PERCENTAGE, UnitOfTime
from homeassistant.core import HomeAssistant
from homeassistant.helpers.device_registry import DeviceInfo
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.helpers.update_coordinator import CoordinatorEntity

from .const import DOMAIN
from .coordinator import SolarLensCoordinator


async def async_setup_entry(
    hass: HomeAssistant, entry: ConfigEntry, async_add_entities: AddEntitiesCallback
) -> None:
    """Set up Solar Lens sensors from a config entry."""
    coordinator: SolarLensCoordinator = hass.data[DOMAIN][entry.entry_id]

    sensors = [
        SolarLensBatteryRemaining(coordinator),
        SolarLensBatteryEmptyTime(coordinator),
        SolarLensChargeResumeTime(coordinator),
        SolarLensGapHours(coordinator),
        SolarLensTomorrowSunsetSoCEstimate(coordinator),
        SolarLensPredictionCurve(coordinator),
    ]

    async_add_entities(sensors)


class SolarLensBaseSensor(CoordinatorEntity[SolarLensCoordinator], SensorEntity):
    """Base class for all Solar Lens sensors."""

    _attr_has_entity_name = True

    def __init__(self, coordinator: SolarLensCoordinator, name: str, key: str) -> None:
        """Initialize the sensor."""
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


class SolarLensBatteryRemaining(SolarLensBaseSensor):
    """Sensor showing estimated hours of battery remaining."""

    _attr_native_unit_of_measurement = UnitOfTime.HOURS
    _attr_icon = "mdi:battery-clock"

    def __init__(self, coordinator: SolarLensCoordinator) -> None:
        """Initialize the sensor."""
        super().__init__(coordinator, "Battery Hours Remaining", "battery_hours_remaining")

    @property
    def native_value(self) -> float | None:
        """Return the estimated battery hours remaining."""
        if not self.coordinator.data:
            return None
        val = self.coordinator.data.get("battery_hours_remaining")
        return float(val) if val is not None else None


class SolarLensBatteryEmptyTime(SolarLensBaseSensor):
    """Sensor showing the timestamp when the battery is predicted to hit the SoC floor."""

    _attr_device_class = SensorDeviceClass.TIMESTAMP
    _attr_icon = "mdi:battery-alert"

    def __init__(self, coordinator: SolarLensCoordinator) -> None:
        """Initialize the sensor."""
        super().__init__(coordinator, "Battery Empty Time", "battery_empty_time")

    @property
    def native_value(self) -> Any | None:
        """Return the predicted empty timestamp."""
        if not self.coordinator.data:
            return None
        return self.coordinator.data.get("battery_empty_time")


class SolarLensChargeResumeTime(SolarLensBaseSensor):
    """Sensor showing the timestamp when solar charging is predicted to resume."""

    _attr_device_class = SensorDeviceClass.TIMESTAMP
    _attr_icon = "mdi:battery-charging-outline"

    def __init__(self, coordinator: SolarLensCoordinator) -> None:
        """Initialize the sensor."""
        super().__init__(coordinator, "Charge Resume Time", "charge_resume_time")

    @property
    def native_value(self) -> Any | None:
        """Return the predicted charge resume timestamp."""
        if not self.coordinator.data:
            return None
        return self.coordinator.data.get("charge_resume_time")


class SolarLensGapHours(SolarLensBaseSensor):
    """Sensor showing the gap in hours between battery emptying and charge resuming."""

    _attr_native_unit_of_measurement = UnitOfTime.HOURS
    _attr_icon = "mdi:clock-alert-outline"

    def __init__(self, coordinator: SolarLensCoordinator) -> None:
        """Initialize the sensor."""
        super().__init__(coordinator, "Gap Hours", "gap_hours")

    @property
    def native_value(self) -> float | None:
        """Return the gap hours value."""
        if not self.coordinator.data:
            return None
        val = self.coordinator.data.get("gap_hours")
        return float(val) if val is not None else None


class SolarLensTomorrowSunsetSoCEstimate(SolarLensBaseSensor):
    """Sensor showing the estimated battery SoC at tomorrow's sunset."""

    _attr_native_unit_of_measurement = PERCENTAGE
    _attr_icon = "mdi:battery-sun"

    def __init__(self, coordinator: SolarLensCoordinator) -> None:
        """Initialize the sensor."""
        super().__init__(
            coordinator,
            "Tomorrow Sunset SoC Estimate",
            "tomorrow_sunset_soc_estimate",
        )

    @property
    def native_value(self) -> float | None:
        """Return the predicted SoC at tomorrow's sunset."""
        if not self.coordinator.data:
            return None
        val = self.coordinator.data.get("tomorrow_sunset_soc_estimate")
        return float(val) if val is not None else None


class SolarLensPredictionCurve(SolarLensBaseSensor):
    """Sensor exposing the 48-hour predicted SoC curve in attributes."""

    _attr_icon = "mdi:chart-timeline-variant"

    def __init__(self, coordinator: SolarLensCoordinator) -> None:
        """Initialize the sensor."""
        super().__init__(coordinator, "Prediction Curve", "prediction_curve")

    @property
    def native_value(self) -> int | None:
        """Return the number of points in the prediction curve."""
        if not self.coordinator.data or "prediction_curve" not in self.coordinator.data:
            return None
        return len(self.coordinator.data["prediction_curve"])

    @property
    def extra_state_attributes(self) -> dict[str, Any] | None:
        """Return the predicted SoC curve as attributes for ApexCharts."""
        if not self.coordinator.data:
            return None
        return {
            "prediction_curve": self.coordinator.data.get("prediction_curve"),
            "predicted_consumption": self.coordinator.data.get("predicted_consumption"),
            "predicted_solar": self.coordinator.data.get("predicted_solar"),
        }
