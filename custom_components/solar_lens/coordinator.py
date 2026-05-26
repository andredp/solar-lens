"""Data update coordinator for Solar Lens integration."""

import logging
from datetime import datetime, timedelta
from typing import Any

import numpy as np
from homeassistant.components.recorder.history import state_changes_during_period
from homeassistant.components.recorder.statistics import statistics_during_period
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant, State
from homeassistant.helpers.update_coordinator import DataUpdateCoordinator, UpdateFailed
from homeassistant.util import dt as dt_util
from sklearn.ensemble import HistGradientBoostingRegressor

from .ml.feature_engineering import compute_rolling_mean, extract_time_features
from .ml.simulator import SolarBatterySimulator

_LOGGER = logging.getLogger(__name__)


class SolarLensCoordinator(DataUpdateCoordinator[dict[str, Any]]):
    """Coordinator to fetch data, retrain model, and run battery simulation."""

    def __init__(self, hass: HomeAssistant, entry: ConfigEntry) -> None:
        """Initialize the coordinator."""
        super().__init__(
            hass,
            _LOGGER,
            config_entry=entry,
            name="Solar Lens Coordinator",
            update_interval=timedelta(minutes=5),
        )
        self.entry = entry

        # Entity configuration from config entry
        self.soc_entity_id = entry.data["soc_entity"]
        self.consumption_entity_id = entry.data["consumption_entity"]
        self.solar_forecast_entity_id = entry.data["solar_forecast_entity"]
        self.weather_entity_id = entry.data["weather_entity"]
        self.battery_capacity_kwh = float(entry.data["battery_capacity_kwh"])

        # ML model and training state
        self.model: HistGradientBoostingRegressor | None = None
        self.last_trained: datetime | None = None
        self.recent_consumption_baseline = 0.5  # fallback kW

        # Initialize physical simulator (default efficiency: 95%, min SoC: 10%)
        self.simulator = SolarBatterySimulator(
            battery_capacity_kwh=self.battery_capacity_kwh,
            charge_efficiency=0.95,
            discharge_efficiency=0.95,
            min_soc=10.0,
        )

    async def _async_update_data(self) -> dict[str, Any]:
        """Fetch data, retrain model if needed, and run simulation."""
        try:
            # 1. Get current battery SoC
            soc_state = self.hass.states.get(self.soc_entity_id)
            if not soc_state or soc_state.state in (None, "unknown", "unavailable"):
                raise UpdateFailed(f"SoC entity {self.soc_entity_id} not available")
            current_soc = float(soc_state.state)

            # 2. Parse solar forecast
            solar_forecast_state = self.hass.states.get(self.solar_forecast_entity_id)
            solar_forecast = self._parse_solar_forecast(solar_forecast_state)

            # 3. Check if we need to retrain the consumption model
            now = dt_util.utcnow()
            if (
                self.model is None
                or self.last_trained is None
                or (now - self.last_trained) > timedelta(hours=1)
            ):
                _LOGGER.info("Starting background retraining of consumption model")
                await self._async_retrain_model()

            # 4. Generate 48h simulation grid (5-min intervals)
            grid_5min = [now + timedelta(minutes=5 * i) for i in range(288 * 2)]
            grid_timestamps = np.array(grid_5min)

            # Fetch and interpolate temperature forecast
            temp_forecast = await self._async_get_temperature_forecast()
            predicted_temp = self._interpolate_temperature_forecast(temp_forecast, grid_timestamps)

            # Predict consumption baseline for each grid step
            predicted_consumption = self._predict_consumption(grid_5min, predicted_temp)

            # Interpolate solar forecast to 5-min grid
            predicted_solar = self._interpolate_solar_forecast(solar_forecast, grid_timestamps)

            # 5. Run forward physical simulation
            sim_results = self.simulator.run_simulation(
                initial_soc=current_soc,
                solar_forecast=predicted_solar,
                consumption_forecast=predicted_consumption,
            )

            # 6. Calculate sensors based on simulation curves
            soc_curve = sim_results["soc"]
            sim_results["wasted_solar"]
            sim_results["grid_import"]

            # battery_empty_time & hours_remaining
            battery_empty_idx = -1
            for idx, val in enumerate(soc_curve):
                if val <= self.simulator.min_soc + 0.1:
                    battery_empty_idx = idx
                    break

            if battery_empty_idx != -1:
                empty_time = grid_5min[battery_empty_idx]
                hours_remaining = float(battery_empty_idx * 5.0 / 60.0)
            else:
                empty_time = None
                hours_remaining = 48.0

            # charge_resume_time
            # Find the first time solar exceeds consumption after battery empties
            # (or starting from now if battery is currently empty)
            charge_resume_time = None
            start_search_idx = max(0, battery_empty_idx)
            if battery_empty_idx != -1 or current_soc <= self.simulator.min_soc + 0.1:
                for idx in range(start_search_idx, len(grid_5min)):
                    if predicted_solar[idx] > predicted_consumption[idx] + 0.05:
                        charge_resume_time = grid_5min[idx]
                        break

            # gap_hours
            gap_hours = 0.0
            if empty_time and charge_resume_time:
                gap_hours = float((charge_resume_time - empty_time).total_seconds() / 3600.0)

            # tomorrow_sunset_soc_estimate
            # Estimate tomorrow's sunset: find when solar production drops to 0 tomorrow afternoon
            tomorrow_sunset_idx = -1
            tomorrow_date = (now + timedelta(days=1)).date()
            # Look at steps falling on tomorrow between 3 PM (15:00) and 10 PM (22:00)
            for idx, ts in enumerate(grid_5min):
                if (
                    ts.date() == tomorrow_date
                    and 15 <= ts.hour <= 22
                    and predicted_solar[idx] > 0.01
                ):
                    tomorrow_sunset_idx = idx

            tomorrow_sunset_soc = 100.0
            if tomorrow_sunset_idx != -1 and tomorrow_sunset_idx < len(soc_curve):
                tomorrow_sunset_soc = soc_curve[tomorrow_sunset_idx]
            else:
                # Fallback: predicted SoC 24 hours from now
                tomorrow_sunset_soc = soc_curve[288]

            # will_battery_last_the_night
            # Battery lasts the night if it doesn't hit empty floor before tomorrow's charge resume
            will_battery_last = not (
                empty_time is not None
                and (charge_resume_time is None or empty_time < charge_resume_time)
            )

            # Format prediction curve for ApexCharts
            prediction_curve = [
                {"timestamp": ts.isoformat(), "soc": round(soc, 1)}
                for ts, soc in zip(grid_5min, soc_curve, strict=True)
            ]

            return {
                "battery_hours_remaining": round(hours_remaining, 1),
                "battery_empty_time": empty_time,
                "charge_resume_time": charge_resume_time,
                "gap_hours": round(gap_hours, 1),
                "tomorrow_sunset_soc_estimate": round(tomorrow_sunset_soc, 1),
                "will_battery_last_the_night": will_battery_last,
                "prediction_curve": prediction_curve,
                "predicted_consumption": predicted_consumption,
                "predicted_solar": predicted_solar,
            }

        except Exception as err:
            raise UpdateFailed(f"Error updating solar prediction data: {err}") from err

    async def _async_retrain_model(self) -> None:
        """Fetch historical statistics and train the model in the background executor."""
        now = dt_util.utcnow()
        start_time = now - timedelta(days=90)  # load up to 90 days of statistics

        # Fetch hourly statistics for house consumption (and temperature sensor if it is a sensor)
        stat_ids = {self.consumption_entity_id}
        is_temp_sensor = self.weather_entity_id.startswith("sensor.")
        if is_temp_sensor:
            stat_ids.add(self.weather_entity_id)

        stats_dict = await self.hass.async_add_executor_job(
            statistics_during_period,
            self.hass,
            start_time,
            None,
            stat_ids,
            "hour",
            None,
            {"mean", "sum", "state"},
        )

        consumption_rows = stats_dict.get(self.consumption_entity_id, [])
        if not consumption_rows:
            _LOGGER.warning(
                "No statistics found for consumption entity %s. Using fallback baseline.",
                self.consumption_entity_id,
            )
            self._train_fallback_model()
            return

        # Parse consumption data into {hour_ts: value}
        consumption_data = {}
        for r in consumption_rows:
            val = self._parse_stat_value(r)
            if val is not None:
                ts = self._parse_stat_start(r["start"])
                hour_ts = ts.replace(minute=0, second=0, microsecond=0)
                consumption_data[hour_ts] = val

        # Parse temperature data
        temperature_data = {}
        if is_temp_sensor:
            temp_rows = stats_dict.get(self.weather_entity_id, [])
            for r in temp_rows:
                val = self._parse_stat_value(r)
                if val is not None:
                    ts = self._parse_stat_start(r["start"])
                    hour_ts = ts.replace(minute=0, second=0, microsecond=0)
                    temperature_data[hour_ts] = val
        else:
            # Query recorder history for the weather entity
            temp_history = await self.hass.async_add_executor_job(
                state_changes_during_period,
                self.hass,
                start_time,
                None,
                self.weather_entity_id,
            )
            history_rows = temp_history.get(self.weather_entity_id, [])
            for state in history_rows:
                if state.state not in (None, "unknown", "unavailable"):
                    temp_val = state.attributes.get("temperature")
                    if temp_val is not None:
                        hour_ts = state.last_updated.replace(minute=0, second=0, microsecond=0)
                        temperature_data[hour_ts] = float(temp_val)

        # Align both time series
        aligned_timestamps = []
        aligned_consumption = []
        aligned_temperature = []

        for hour_ts in sorted(consumption_data.keys()):
            if hour_ts in temperature_data:
                aligned_timestamps.append(hour_ts)
                aligned_consumption.append(consumption_data[hour_ts])
                aligned_temperature.append(temperature_data[hour_ts])

        if len(aligned_timestamps) < 24:
            _LOGGER.warning("Insufficient aligned history to train ML model. Using fallback.")
            self._train_fallback_model()
            return

        timestamps = aligned_timestamps
        values = np.array(aligned_consumption)
        temps = np.array(aligned_temperature)

        # Compute rolling mean over past 24 hours to use as recent baseline feature
        roll_mean_24 = compute_rolling_mean(values, 24)
        roll_mean_24_shifted = np.roll(roll_mean_24, 1)
        roll_mean_24_shifted[0] = roll_mean_24[0]

        # Extract features
        time_feats = extract_time_features(timestamps)
        X = np.column_stack(
            [
                time_feats["time_sin"],
                time_feats["time_cos"],
                time_feats["day_sin"],
                time_feats["day_cos"],
                time_feats["day_of_week"],
                time_feats["is_weekend"],
                roll_mean_24_shifted,
                temps,
            ]
        )
        y = values

        # Train model
        model = HistGradientBoostingRegressor(random_state=42, max_iter=50)
        await self.hass.async_add_executor_job(model.fit, X, y)

        self.model = model
        self.last_trained = now
        # Update current recent baseline (average consumption of the last 24h)
        self.recent_consumption_baseline = float(roll_mean_24[-1])
        _LOGGER.info(
            "Successfully trained consumption model on %d samples. Current baseline: %.2f kW",
            len(y),
            self.recent_consumption_baseline,
        )

    def _train_fallback_model(self) -> None:
        """Initialize a fallback model that returns static average patterns."""
        self.model = None
        self.last_trained = dt_util.utcnow()
        # Fallback average consumption
        self.recent_consumption_baseline = 0.5

    def _predict_consumption(
        self, datetimes: list[datetime], temperatures: list[float]
    ) -> list[float]:
        """Predict hourly consumption for the list of future datetimes and temperatures."""
        if self.model is None:
            # Fallback static diurnal profile if no model is trained
            predictions = []
            for dt, temp in zip(datetimes, temperatures, strict=True):
                # Diurnal profile: peaks morning (8:00) and evening (19:00)
                hour = dt.hour + dt.minute / 60.0
                morning_peak = 0.8 * np.exp(-0.5 * ((hour - 8.0) / 1.0) ** 2)
                evening_peak = 1.5 * np.exp(-0.5 * ((hour - 19.0) / 1.5) ** 2)
                val = self.recent_consumption_baseline + morning_peak + evening_peak
                # HVAC adjustment based on temp (heating below 15°C, cooling above 25°C)
                if temp < 15.0:
                    val += 0.08 * (15.0 - temp)
                elif temp > 25.0:
                    val += 0.12 * (temp - 25.0)
                if dt.weekday() >= 5:  # weekend boost
                    val *= 1.15
                predictions.append(max(0.05, float(val)))
            return predictions

        # Prepare features for the prediction grid
        time_feats = extract_time_features(datetimes)
        recent_baseline_feature = np.full(len(datetimes), self.recent_consumption_baseline)

        X_pred = np.column_stack(
            [
                time_feats["time_sin"],
                time_feats["time_cos"],
                time_feats["day_sin"],
                time_feats["day_cos"],
                time_feats["day_of_week"],
                time_feats["is_weekend"],
                recent_baseline_feature,
                temperatures,
            ]
        )

        preds = self.model.predict(X_pred)
        return [max(0.05, float(p)) for p in preds]

    async def _async_get_temperature_forecast(self) -> list[tuple[datetime, float]]:
        """Fetch future temperature forecast (datetime, °C) from weather/temperature entity."""
        forecast_list: list[tuple[datetime, float]] = []

        # 1. Try calling the modern service weather.get_forecasts
        try:
            response = await self.hass.services.async_call(
                "weather",
                "get_forecasts",
                {"entity_id": self.weather_entity_id, "type": "hourly"},
                blocking=True,
                return_response=True,
            )
            if response and self.weather_entity_id in response:
                forecast_data = response[self.weather_entity_id].get("forecast", [])
                for item in forecast_data:
                    ts_val = item.get("datetime")
                    temp_val = item.get("temperature")
                    if ts_val is not None and temp_val is not None:
                        ts = dt_util.parse_datetime(str(ts_val))
                        if ts:
                            forecast_list.append((ts, float(temp_val)))
                if forecast_list:
                    return forecast_list
        except Exception as err:
            _LOGGER.debug("Failed to call weather.get_forecasts service: %s", err)

        # 2. Fallback: check weather entity state and attributes
        state = self.hass.states.get(self.weather_entity_id)
        if state:
            attrs = state.attributes
            if "forecast" in attrs and isinstance(attrs["forecast"], list):
                for item in attrs["forecast"]:
                    ts_val = item.get("datetime")
                    temp_val = item.get("temperature")
                    if ts_val is not None and temp_val is not None:
                        ts = dt_util.parse_datetime(str(ts_val))
                        if ts:
                            forecast_list.append((ts, float(temp_val)))

        # 3. Fallback: if direct temp sensor (no forecast), assume constant current temp
        if not forecast_list and state:
            try:
                current_temp = float(state.state)
                now = dt_util.utcnow()
                forecast_list = [(now + timedelta(hours=i), current_temp) for i in range(49)]
            except ValueError:
                pass

        return forecast_list

    def _interpolate_temperature_forecast(
        self, forecast_list: list[tuple[datetime, float]], target_grid: np.ndarray
    ) -> list[float]:
        """Interpolate hourly temperature forecast into the target 5-minute grid."""
        if not forecast_list:
            return [20.0] * len(target_grid)

        forecast_times = np.array([ts.timestamp() for ts, _ in forecast_list])
        forecast_values = np.array([val for _, val in forecast_list])

        grid_times = np.array([ts.timestamp() for ts in target_grid])

        # Run linear interpolation, fill using edge values outside range
        interpolated = np.interp(
            grid_times,
            forecast_times,
            forecast_values,
            left=forecast_values[0],
            right=forecast_values[-1],
        )
        return [float(v) for v in interpolated]

    def _parse_solar_forecast(self, state: State | None) -> list[tuple[datetime, float]]:
        """Parse future solar forecasts (datetime, kW) from forecast entity state."""
        forecast_list: list[tuple[datetime, float]] = []
        if not state:
            return forecast_list

        attrs = state.attributes

        # 1. Check if standard forecasts list exists (e.g. Forecast.Solar or Solcast)
        if "forecasts" in attrs and isinstance(attrs["forecasts"], list):
            for item in attrs["forecasts"]:
                ts_val = item.get("period_start") or item.get("datetime") or item.get("time")
                power_val = item.get("pv_estimate") or item.get("power") or item.get("value")
                if ts_val is not None and power_val is not None:
                    ts = dt_util.parse_datetime(str(ts_val))
                    if ts:
                        forecast_list.append((ts, float(power_val)))

        # 2. Check for Forecast.Solar dict format in "watt_hours" or "wh_period"
        elif "watt_hours" in attrs and isinstance(attrs["watt_hours"], dict):
            for key, val in attrs["watt_hours"].items():
                ts = dt_util.parse_datetime(str(key))
                if ts:
                    # Convert Wh (hourly energy) to kW (average hourly power)
                    forecast_list.append((ts, float(val) / 1000.0))

        elif "wh_period" in attrs and isinstance(attrs["wh_period"], dict):
            for key, val in attrs["wh_period"].items():
                ts = dt_util.parse_datetime(str(key))
                if ts:
                    forecast_list.append((ts, float(val) / 1000.0))

        # 3. Check for standard weather forecast attribute
        elif "forecast" in attrs and isinstance(attrs["forecast"], list):
            # Weather forecasts usually don't have fine solar forecasts, but we check just in case
            for item in attrs["forecast"]:
                ts_val = item.get("datetime")
                power_val = item.get("solar_production")  # some custom weather components
                if ts_val is not None and power_val is not None:
                    ts = dt_util.parse_datetime(str(ts_val))
                    if ts:
                        forecast_list.append((ts, float(power_val)))

        # Sort chronological
        forecast_list.sort(key=lambda x: x[0])
        return forecast_list

    def _interpolate_solar_forecast(
        self, forecast_list: list[tuple[datetime, float]], target_grid: np.ndarray
    ) -> list[float]:
        """Interpolate parsed hourly solar forecast into the target 5-minute grid."""
        if not forecast_list:
            return [0.0] * len(target_grid)

        forecast_times = np.array([ts.timestamp() for ts, _ in forecast_list])
        forecast_values = np.array([val for _, val in forecast_list])

        grid_times = np.array([ts.timestamp() for ts in target_grid])

        # Run linear interpolation, fill 0.0 outside forecast range (e.g. night or past)
        interpolated = np.interp(grid_times, forecast_times, forecast_values, left=0.0, right=0.0)
        return [max(0.0, float(v)) for v in interpolated]

    def _parse_stat_start(self, start_val: Any) -> datetime:
        """Parse statistics row start timestamp or datetime."""
        if isinstance(start_val, datetime):
            return start_val
        if isinstance(start_val, (int, float)):
            return datetime.fromtimestamp(start_val, tz=dt_util.UTC)
        if isinstance(start_val, str):
            dt = dt_util.parse_datetime(start_val)
            if dt:
                return dt
        raise ValueError(f"Cannot parse start value {start_val}")

    def _parse_stat_value(self, row: dict[str, Any]) -> float | None:
        """Parse numerical value from statistics row preferring mean, then sum, then state."""
        for key in ("mean", "sum", "state"):
            if row.get(key) is not None:
                return float(row[key])
        return None
