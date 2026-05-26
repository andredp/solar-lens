"""Data update coordinator for Solar Lens integration."""

import asyncio
import contextlib
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
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeRegressor

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

        self.actual_solar_entity_id = entry.data.get("actual_solar_entity")
        self.charge_limit_entity_id = entry.data.get("charge_limit_entity")
        self.discharge_limit_entity_id = entry.data.get("discharge_limit_entity")
        self.battery_voltage_entity_id = entry.data.get("battery_voltage_entity")
        self.battery_temp_entity_id = entry.data.get("battery_temp_entity")

        # ML models and training state
        self.consumption_model: HistGradientBoostingRegressor | None = None
        self.solar_model: HistGradientBoostingRegressor | None = None
        self.temp_model: Ridge | None = None
        self.charge_limit_model: DecisionTreeRegressor | None = None
        self.discharge_limit_model: DecisionTreeRegressor | None = None

        self.last_trained: datetime | None = None
        self._training_task: asyncio.Task[None] | None = None
        self.recent_consumption_baseline = 0.5  # fallback kW

        # Initialize physical simulator (default efficiency: 95%, min SoC: 10%)
        self.simulator = SolarBatterySimulator(
            battery_capacity_kwh=self.battery_capacity_kwh,
            charge_efficiency=0.95,
            discharge_efficiency=0.95,
            min_soc=10.0,
        )

    def _to_celsius(self, temp: float, unit: str | None) -> float:
        """Convert Fahrenheit to Celsius if needed."""
        if unit in ("°F", "F"):
            return (temp - 32.0) * 5.0 / 9.0
        return temp

    async def _async_update_data(self) -> dict[str, Any]:
        """Fetch data, trigger background training, and run simulation."""
        try:
            # 1. Get current battery SoC
            soc_state = self.hass.states.get(self.soc_entity_id)
            if not soc_state or soc_state.state in (None, "unknown", "unavailable"):
                raise UpdateFailed(f"SoC entity {self.soc_entity_id} not available")
            current_soc = float(soc_state.state)

            # 2. Parse solar forecast
            solar_forecast_state = self.hass.states.get(self.solar_forecast_entity_id)
            solar_forecast = self._parse_solar_forecast(solar_forecast_state)

            # 3. Trigger background retraining if needed
            now = dt_util.utcnow()
            if (self.last_trained is None or (now - self.last_trained) > timedelta(hours=1)) and (
                not hasattr(self, "_training_task")
                or self._training_task is None
                or self._training_task.done()
            ):
                _LOGGER.info("Starting background retraining of ML models")
                self._training_task = self.hass.async_create_background_task(
                    self._async_retrain_models(),
                    "solar_lens_retrain_models",
                )

            # 4. Generate local 48h simulation grid (5-min intervals)
            local_now = dt_util.as_local(now)
            grid_5min = [local_now + timedelta(minutes=5 * i) for i in range(288 * 2)]
            grid_timestamps = np.array(grid_5min)

            # Fetch temperature forecast and convert to Celsius
            temp_forecast = await self._async_get_temperature_forecast()
            temp_forecast.sort(key=lambda x: x[0])

            # Detect temperature unit
            is_f = False
            weather_state = self.hass.states.get(self.weather_entity_id)
            if weather_state:
                attrs = weather_state.attributes
                unit = attrs.get("temperature_unit") or attrs.get("unit_of_measurement")
                if unit in ("°F", "F"):
                    is_f = True

            predicted_temp_raw = self._interpolate_temperature_forecast(
                temp_forecast, grid_timestamps
            )
            predicted_temp = [
                self._to_celsius(t, "F" if is_f else None) for t in predicted_temp_raw
            ]

            # Predict consumption baseline
            predicted_consumption = self._predict_consumption(grid_5min, predicted_temp)

            # Interpolate raw solar forecast and refine
            raw_solar_5min = self._interpolate_solar_forecast(solar_forecast, grid_timestamps)
            predicted_solar = self._predict_solar(grid_5min, raw_solar_5min)

            # Read live voltage and battery temp
            live_voltage = 48.0
            if self.battery_voltage_entity_id:
                volt_state = self.hass.states.get(self.battery_voltage_entity_id)
                if volt_state and volt_state.state not in (None, "unknown", "unavailable"):
                    with contextlib.suppress(ValueError):
                        live_voltage = float(volt_state.state)

            live_bat_temp = predicted_temp[0]
            if self.battery_temp_entity_id:
                btemp_state = self.hass.states.get(self.battery_temp_entity_id)
                if btemp_state and btemp_state.state not in (None, "unknown", "unavailable"):
                    try:
                        btemp_unit = btemp_state.attributes.get("unit_of_measurement")
                        live_bat_temp = self._to_celsius(float(btemp_state.state), btemp_unit)
                    except ValueError:
                        pass

            # 5. Run sequential physical simulation loop with ML model clamping
            soc_curve = []
            wasted_solar_curve = []
            grid_import_curve = []
            net_flow_curve = []

            battery_energy = (current_soc / 100.0) * self.battery_capacity_kwh
            dt = 5.0 / 60.0
            current_soc_sim = current_soc
            current_bat_temp_sim = live_bat_temp

            for i in range(len(grid_5min)):
                solar = predicted_solar[i]
                consumption = predicted_consumption[i]
                out_temp = predicted_temp[i]

                # Predict battery temperature using unclamped net flow proxy
                diff = solar - consumption
                if diff > 0:
                    est_net_flow = diff * self.simulator.charge_efficiency
                else:
                    est_net_flow = diff / self.simulator.discharge_efficiency

                if self.temp_model is not None:
                    try:
                        X_t = np.array([[out_temp, abs(est_net_flow), est_net_flow**2]])
                        current_bat_temp_sim = float(self.temp_model.predict(X_t)[0])
                    except Exception:
                        current_bat_temp_sim = out_temp
                else:
                    current_bat_temp_sim = 0.9 * current_bat_temp_sim + 0.1 * out_temp

                # Predict BMS charge/discharge limit (Amps)
                if self.charge_limit_model is not None:
                    try:
                        pred_charge_limit_a = float(
                            self.charge_limit_model.predict(
                                [[current_soc_sim, current_bat_temp_sim]]
                            )[0]
                        )
                    except Exception:
                        pred_charge_limit_a = 75.0
                else:
                    pred_charge_limit_a = 75.0

                if self.discharge_limit_model is not None:
                    try:
                        pred_discharge_limit_a = float(
                            self.discharge_limit_model.predict(
                                [[current_soc_sim, current_bat_temp_sim]]
                            )[0]
                        )
                    except Exception:
                        pred_discharge_limit_a = 75.0
                else:
                    pred_discharge_limit_a = 75.0

                # Convert limits (Amps) to power (kW), clamping limits to non-negative values
                max_charge_power_kw = (max(0.0, pred_charge_limit_a) * live_voltage) / 1000.0
                max_discharge_power_kw = (max(0.0, pred_discharge_limit_a) * live_voltage) / 1000.0

                # Step simulation
                (
                    battery_energy,
                    net_flow_val,
                    wasted,
                    imported,
                ) = self.simulator.update_battery_state(
                    battery_energy,
                    solar,
                    consumption,
                    dt,
                    max_charge_power_kw,
                    max_discharge_power_kw,
                )

                current_soc_sim = (battery_energy / self.battery_capacity_kwh) * 100.0

                soc_curve.append(float(current_soc_sim))
                wasted_solar_curve.append(float(wasted))
                grid_import_curve.append(float(imported))
                net_flow_curve.append(float(net_flow_val))

            # 6. Calculate sensors based on simulation curves
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
            tomorrow_sunset_idx = -1
            tomorrow_date = (local_now + timedelta(days=1)).date()
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

    async def _async_fetch_hourly_data(
        self, entity_id: str, start_time: datetime, default_val: float
    ) -> dict[datetime, float]:
        """Fetch hourly history/statistics for an entity and return {local_hour_ts: value}."""
        data: dict[datetime, float] = {}

        # 1. Try statistics first
        try:
            stats_dict = await self.hass.async_add_executor_job(
                statistics_during_period,
                self.hass,
                start_time,
                None,
                {entity_id},
                "hour",
                None,
                {"mean", "sum", "state"},
            )
            rows = stats_dict.get(entity_id, [])
            for r in rows:
                val = self._parse_stat_value(r)
                if val is not None:
                    ts = self._parse_stat_start(r["start"])
                    local_ts = dt_util.as_local(ts)
                    hour_ts = local_ts.replace(minute=0, second=0, microsecond=0)
                    data[hour_ts] = val

            # Ensure we have a reasonable amount of statistics data, otherwise fall back to recorder
            if len(data) >= 24:
                return data
        except Exception as err:
            _LOGGER.debug("Statistics not available for %s: %s", entity_id, err)

        # 2. Fallback to state changes from recorder for the last 10 days
        try:
            recorder_start = max(start_time, dt_util.utcnow() - timedelta(days=10))
            history_dict = await self.hass.async_add_executor_job(
                state_changes_during_period,
                self.hass,
                recorder_start,
                None,
                entity_id,
            )
            rows = history_dict.get(entity_id, [])

            hourly_sums: dict[datetime, float] = {}
            hourly_counts: dict[datetime, int] = {}
            is_weather_domain = entity_id.startswith("weather.")

            for state in rows:
                if state.state not in (None, "unknown", "unavailable"):
                    try:
                        if is_weather_domain:
                            val = state.attributes.get("temperature")
                            if val is None:
                                continue
                            val = float(val)
                        else:
                            val = float(state.state)
                        local_ts = dt_util.as_local(state.last_updated)
                        hour_ts = local_ts.replace(minute=0, second=0, microsecond=0)
                        hourly_sums[hour_ts] = hourly_sums.get(hour_ts, 0.0) + val
                        hourly_counts[hour_ts] = hourly_counts.get(hour_ts, 0) + 1
                    except ValueError:
                        continue

            for hour_ts, total in hourly_sums.items():
                data[hour_ts] = total / hourly_counts[hour_ts]
        except Exception as err:
            _LOGGER.warning("Failed to fetch history for %s: %s", entity_id, err)

        return data

    def _align_and_fill(
        self,
        grid: list[datetime],
        raw_data: dict[datetime, float],
        default_value: float,
    ) -> np.ndarray:
        """Align raw_data dictionary to grid timestamps using forward-fill."""
        values = []
        last_val = None
        sorted_keys = sorted(raw_data.keys())

        for ts in grid:
            if ts in raw_data:
                last_val = raw_data[ts]
            elif last_val is None:
                # Backward-fill from first available future key
                future_keys = [k for k in sorted_keys if k > ts]
                last_val = raw_data[future_keys[0]] if future_keys else default_value
            values.append(last_val)
        return np.array(values)

    async def _async_retrain_models(self) -> None:
        """Fetch historical statistics and train all 5 ML models in the background."""
        now = dt_util.utcnow()
        start_time = now - timedelta(days=90)

        # Build list of async fetch tasks
        fetch_tasks = [
            self._async_fetch_hourly_data(self.consumption_entity_id, start_time, 0.5),
            self._async_fetch_hourly_data(self.weather_entity_id, start_time, 20.0),
            self._async_fetch_hourly_data(self.soc_entity_id, start_time, 50.0),
        ]

        opt_keys = []
        if self.actual_solar_entity_id:
            fetch_tasks.append(
                self._async_fetch_hourly_data(self.actual_solar_entity_id, start_time, 0.0)
            )
            opt_keys.append("actual_solar")

        if self.battery_temp_entity_id:
            fetch_tasks.append(
                self._async_fetch_hourly_data(self.battery_temp_entity_id, start_time, 20.0)
            )
            opt_keys.append("battery_temp")

        if self.charge_limit_entity_id:
            fetch_tasks.append(
                self._async_fetch_hourly_data(self.charge_limit_entity_id, start_time, 75.0)
            )
            opt_keys.append("charge_limit")

        if self.discharge_limit_entity_id:
            fetch_tasks.append(
                self._async_fetch_hourly_data(self.discharge_limit_entity_id, start_time, 75.0)
            )
            opt_keys.append("discharge_limit")

        fetch_tasks.append(
            self._async_fetch_hourly_data(self.solar_forecast_entity_id, start_time, 0.0)
        )
        opt_keys.append("solar_forecast")

        try:
            results = await asyncio.gather(*fetch_tasks)
        except Exception as err:
            _LOGGER.error("Failed to fetch historical data for model retraining: %s", err)
            self.last_trained = now
            return

        try:
            # Map results
            consumption_raw = results[0]
            weather_raw = results[1]
            soc_raw = results[2]

            idx = 3
            actual_solar_raw = {}
            if "actual_solar" in opt_keys:
                actual_solar_raw = results[idx]
                idx += 1

            battery_temp_raw = {}
            if "battery_temp" in opt_keys:
                battery_temp_raw = results[idx]
                idx += 1

            charge_limit_raw = {}
            if "charge_limit" in opt_keys:
                charge_limit_raw = results[idx]
                idx += 1

            discharge_limit_raw = {}
            if "discharge_limit" in opt_keys:
                discharge_limit_raw = results[idx]
                idx += 1

            raw_solar_forecast_raw = {}
            if "solar_forecast" in opt_keys:
                raw_solar_forecast_raw = results[idx]
                idx += 1

            # Align to hourly grid
            grid = sorted(consumption_raw.keys())
            if len(grid) < 24:
                _LOGGER.warning("Insufficient history to train ML models. Retrying later.")
                return

            # Temperature unit normalization
            weather_unit = None
            weather_state = self.hass.states.get(self.weather_entity_id)
            if weather_state:
                weather_unit = weather_state.attributes.get(
                    "temperature_unit"
                ) or weather_state.attributes.get("unit_of_measurement")

            bat_temp_unit = None
            if self.battery_temp_entity_id:
                bat_temp_state = self.hass.states.get(self.battery_temp_entity_id)
                if bat_temp_state:
                    bat_temp_unit = bat_temp_state.attributes.get("unit_of_measurement")

            # Align variables
            consumption = np.array([consumption_raw[ts] for ts in grid])

            outdoor_temp_raw = self._align_and_fill(grid, weather_raw, 20.0)
            outdoor_temp = np.array([self._to_celsius(t, weather_unit) for t in outdoor_temp_raw])

            soc = self._align_and_fill(grid, soc_raw, 50.0)

            actual_solar = (
                self._align_and_fill(grid, actual_solar_raw, 0.0)
                if self.actual_solar_entity_id
                else np.zeros(len(grid))
            )

            battery_temp_raw_aligned = (
                self._align_and_fill(grid, battery_temp_raw, 20.0)
                if self.battery_temp_entity_id
                else outdoor_temp_raw.copy()
            )
            if self.battery_temp_entity_id:
                battery_temp = np.array(
                    [self._to_celsius(t, bat_temp_unit) for t in battery_temp_raw_aligned]
                )
            else:
                battery_temp = outdoor_temp.copy()

            charge_limit = (
                self._align_and_fill(grid, charge_limit_raw, 75.0)
                if self.charge_limit_entity_id
                else np.full(len(grid), 75.0)
            )
            discharge_limit = (
                self._align_and_fill(grid, discharge_limit_raw, 75.0)
                if self.discharge_limit_entity_id
                else np.full(len(grid), 75.0)
            )
            raw_solar_forecast = self._align_and_fill(grid, raw_solar_forecast_raw, 0.0)

            # Calculate historical DC power flow proxy
            hist_diff = actual_solar - consumption
            hist_net_flow = np.where(
                hist_diff > 0,
                hist_diff * self.simulator.charge_efficiency,
                hist_diff / self.simulator.discharge_efficiency,
            )

            # Run training in the executor thread pool
            await self.hass.async_add_executor_job(
                self._train_models,
                grid,
                consumption,
                outdoor_temp,
                soc,
                actual_solar,
                raw_solar_forecast,
                battery_temp,
                charge_limit,
                discharge_limit,
                hist_net_flow,
            )
        except Exception as err:
            _LOGGER.error("Error during ML model training: %s", err)
        finally:
            self.last_trained = now

    def _train_models(
        self,
        grid: list[datetime],
        consumption: np.ndarray,
        outdoor_temp: np.ndarray,
        soc: np.ndarray,
        actual_solar: np.ndarray,
        raw_solar_forecast: np.ndarray,
        battery_temp: np.ndarray,
        charge_limit: np.ndarray,
        discharge_limit: np.ndarray,
        hist_net_flow: np.ndarray,
    ) -> None:
        """Fit all 5 ML models on aligned data."""
        # 1. Consumption Model
        try:
            roll_mean_24 = compute_rolling_mean(consumption, 24)
            roll_mean_24_shifted = np.roll(roll_mean_24, 1)
            roll_mean_24_shifted[0] = roll_mean_24[0]

            time_feats = extract_time_features(grid)
            X_con = np.column_stack(
                [
                    time_feats["time_sin"],
                    time_feats["time_cos"],
                    time_feats["day_sin"],
                    time_feats["day_cos"],
                    time_feats["day_of_week"],
                    time_feats["is_weekend"],
                    roll_mean_24_shifted,
                    outdoor_temp,
                ]
            )

            c_model = HistGradientBoostingRegressor(random_state=42, max_iter=50)
            c_model.fit(X_con, consumption)
            self.consumption_model = c_model
            self.recent_consumption_baseline = float(roll_mean_24[-1])
            _LOGGER.info(
                "Trained consumption model baseline: %.2f kW",
                self.recent_consumption_baseline,
            )
        except Exception as err:
            _LOGGER.error("Failed to train consumption model: %s", err)

        # 2. Solar Forecast Refiner Model
        if self.actual_solar_entity_id:
            try:
                # Censoring: exclude hours where SoC > 98% and actual solar is curtailed
                non_clipped = (soc <= 98.0) | (actual_solar >= raw_solar_forecast - 0.1)
                if np.sum(non_clipped) >= 24:
                    X_sol = np.column_stack(
                        [
                            time_feats["time_sin"][non_clipped],
                            time_feats["time_cos"][non_clipped],
                            time_feats["day_sin"][non_clipped],
                            time_feats["day_cos"][non_clipped],
                            time_feats["day_of_week"][non_clipped],
                            time_feats["is_weekend"][non_clipped],
                            raw_solar_forecast[non_clipped],
                        ]
                    )
                    y_sol = actual_solar[non_clipped]

                    s_model = HistGradientBoostingRegressor(random_state=42, max_iter=50)
                    s_model.fit(X_sol, y_sol)
                    self.solar_model = s_model
                    _LOGGER.info("Successfully trained Solar Refiner model.")
                else:
                    _LOGGER.warning(
                        "Too few non-clipped solar samples. Skipping solar refiner training."
                    )
            except Exception as err:
                _LOGGER.error("Failed to train solar refiner model: %s", err)

        # 3. Battery Temperature Predictor Model
        if self.battery_temp_entity_id:
            try:
                X_temp = np.column_stack(
                    [
                        outdoor_temp,
                        np.abs(hist_net_flow),
                        hist_net_flow**2,
                    ]
                )
                t_model = Ridge(alpha=1.0)
                t_model.fit(X_temp, battery_temp)
                self.temp_model = t_model
                _LOGGER.info("Successfully trained Battery Temperature Predictor.")
            except Exception as err:
                _LOGGER.error("Failed to train battery temperature model: %s", err)

        # 4. BMS Charge Limit Model
        if self.charge_limit_entity_id:
            try:
                X_lim = np.column_stack([soc, battery_temp])
                chg_model = DecisionTreeRegressor(max_depth=4, random_state=42)
                chg_model.fit(X_lim, charge_limit)
                self.charge_limit_model = chg_model
                _LOGGER.info("Successfully trained BMS Charge Limit model.")
            except Exception as err:
                _LOGGER.error("Failed to train BMS charge limit model: %s", err)

        # 5. BMS Discharge Limit Model
        if self.discharge_limit_entity_id:
            try:
                X_lim = np.column_stack([soc, battery_temp])
                dis_model = DecisionTreeRegressor(max_depth=4, random_state=42)
                dis_model.fit(X_lim, discharge_limit)
                self.discharge_limit_model = dis_model
                _LOGGER.info("Successfully trained BMS Discharge Limit model.")
            except Exception as err:
                _LOGGER.error("Failed to train BMS discharge limit model: %s", err)

    def _predict_consumption(
        self, datetimes: list[datetime], temperatures: list[float]
    ) -> list[float]:
        """Predict hourly consumption for the list of future datetimes and temperatures."""
        if self.consumption_model is None:
            # Fallback static diurnal profile
            predictions = []
            for dt, temp in zip(datetimes, temperatures, strict=True):
                hour = dt.hour + dt.minute / 60.0
                morning_peak = 0.8 * np.exp(-0.5 * ((hour - 8.0) / 1.0) ** 2)
                evening_peak = 1.5 * np.exp(-0.5 * ((hour - 19.0) / 1.5) ** 2)
                val = self.recent_consumption_baseline + morning_peak + evening_peak
                if temp < 15.0:
                    val += 0.08 * (15.0 - temp)
                elif temp > 25.0:
                    val += 0.12 * (temp - 25.0)
                if dt.weekday() >= 5:
                    val *= 1.15
                predictions.append(max(0.05, float(val)))
            return predictions

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

        preds = self.consumption_model.predict(X_pred)
        return [max(0.05, float(p)) for p in preds]

    def _predict_solar(self, datetimes: list[datetime], raw_forecast: list[float]) -> list[float]:
        """Predict refined solar production using the trained solar model."""
        if self.solar_model is None:
            return raw_forecast

        try:
            time_feats = extract_time_features(datetimes)
            X_pred = np.column_stack(
                [
                    time_feats["time_sin"],
                    time_feats["time_cos"],
                    time_feats["day_sin"],
                    time_feats["day_cos"],
                    time_feats["day_of_week"],
                    time_feats["is_weekend"],
                    raw_forecast,
                ]
            )
            preds = self.solar_model.predict(X_pred)
            return [max(0.0, float(p)) for p in preds]
        except Exception as err:
            _LOGGER.debug("Solar refiner prediction failed, falling back to raw: %s", err)
            return raw_forecast

    async def _async_get_temperature_forecast(self) -> list[tuple[datetime, float]]:
        """Fetch future temperature forecast (datetime, °C) from weather/temperature entity."""
        forecast_list: list[tuple[datetime, float]] = []

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
        """Interpolate temperature forecast onto the target 5-minute grid."""
        if not forecast_list:
            return [20.0] * len(target_grid)

        forecast_times = np.array([ts.timestamp() for ts, _ in forecast_list])
        forecast_values = np.array([val for _, val in forecast_list])
        grid_times = np.array([ts.timestamp() for ts in target_grid])

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

        if "forecasts" in attrs and isinstance(attrs["forecasts"], list):
            for item in attrs["forecasts"]:
                ts_val = item.get("period_start") or item.get("datetime") or item.get("time")
                power_val = item.get("pv_estimate") or item.get("power") or item.get("value")
                if ts_val is not None and power_val is not None:
                    ts = dt_util.parse_datetime(str(ts_val))
                    if ts:
                        forecast_list.append((ts, float(power_val)))

        elif "watt_hours" in attrs and isinstance(attrs["watt_hours"], dict):
            for key, val in attrs["watt_hours"].items():
                ts = dt_util.parse_datetime(str(key))
                if ts:
                    forecast_list.append((ts, float(val) / 1000.0))

        elif "wh_period" in attrs and isinstance(attrs["wh_period"], dict):
            for key, val in attrs["wh_period"].items():
                ts = dt_util.parse_datetime(str(key))
                if ts:
                    forecast_list.append((ts, float(val) / 1000.0))

        elif "forecast" in attrs and isinstance(attrs["forecast"], list):
            for item in attrs["forecast"]:
                ts_val = item.get("datetime")
                power_val = item.get("solar_production")
                if ts_val is not None and power_val is not None:
                    ts = dt_util.parse_datetime(str(ts_val))
                    if ts:
                        forecast_list.append((ts, float(power_val)))

        forecast_list.sort(key=lambda x: x[0])
        return forecast_list

    def _interpolate_solar_forecast(
        self, forecast_list: list[tuple[datetime, float]], target_grid: np.ndarray
    ) -> list[float]:
        """Interpolate hourly solar forecast into the target 5-minute grid."""
        if not forecast_list:
            return [0.0] * len(target_grid)

        forecast_times = np.array([ts.timestamp() for ts, _ in forecast_list])
        forecast_values = np.array([val for _, val in forecast_list])
        grid_times = np.array([ts.timestamp() for ts in target_grid])

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
