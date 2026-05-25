"""Deterministic Simulator for Solar Lens solar production, house consumption,

and battery dynamics.
"""

import numpy as np
import pandas as pd


class SolarBatterySimulator:
    """Simulates realistic solar panels, household consumption, and battery physics."""

    def __init__(
        self,
        battery_capacity_kwh: float = 7.2,
        max_solar_kw: float = 4.0,
        base_load_kw: float = 0.3,
        charge_efficiency: float = 0.95,
        discharge_efficiency: float = 0.95,
        min_soc: float = 10.0,
        seed: int = 42,
    ) -> None:
        """Initialize the simulator with physical configuration."""
        self.battery_capacity_kwh = battery_capacity_kwh
        self.max_solar_kw = max_solar_kw
        self.base_load_kw = base_load_kw
        self.charge_efficiency = charge_efficiency
        self.discharge_efficiency = discharge_efficiency
        self.min_soc = min_soc
        self.rng = np.random.default_rng(seed)

    def simulate_solar(self, timestamps: pd.DatetimeIndex, cloud_cover: np.ndarray) -> np.ndarray:
        """Simulate solar production based on day of year, hour of day, and cloud cover."""
        solar_generation = []
        for i, ts in enumerate(timestamps):
            day_of_year = ts.dayofyear
            hour = ts.hour + ts.minute / 60.0

            # Seasonality coefficient: peaks in summer (June 21, day 172), lowest in winter (Dec 21)
            # Shifts range of max solar between 30% (winter) and 100% (summer) of max_solar_kw
            season_coef = 0.65 + 0.35 * np.cos(2.0 * np.pi * (day_of_year - 172) / 365.25)

            # Determine sunrise and sunset based on day of year
            # In June, sunrise is ~5:30 (5.5) and sunset is ~20:30 (20.5)
            # In Dec, sunrise is ~8:00 (8.0) and sunset is ~16:30 (16.5)
            sunrise = 6.75 - 1.25 * np.cos(2.0 * np.pi * (day_of_year - 172) / 365.25)
            sunset = 18.5 + 2.0 * np.cos(2.0 * np.pi * (day_of_year - 172) / 365.25)

            if sunrise < hour < sunset:
                # Bell-shaped curve peaking at noon (12.0)
                noon = (sunrise + sunset) / 2.0
                width = (sunset - sunrise) / 2.0
                bell = np.cos(np.pi * (hour - noon) / (width * 2.0)) ** 2

                # Apply solar capacity, seasonal scale, and cloud attenuation
                cloud_attenuation = 1.0 - 0.8 * cloud_cover[i]
                solar = self.max_solar_kw * season_coef * bell * cloud_attenuation
                solar_generation.append(max(0.0, solar))
            else:
                solar_generation.append(0.0)

        return np.array(solar_generation)

    def simulate_consumption(
        self, timestamps: pd.DatetimeIndex, temperatures: np.ndarray
    ) -> np.ndarray:
        """Simulate house consumption with morning/evening peaks and HVAC dependency."""
        consumption = []
        for i, ts in enumerate(timestamps):
            hour = ts.hour + ts.minute / 60.0
            is_weekend = ts.dayofweek >= 5
            temp = temperatures[i]

            # Base load + noise
            load = self.base_load_kw + self.rng.normal(0.0, 0.05)

            # Double peak structure: morning (7:00-9:00) and evening (17:00-21:00)
            # Morning peak (e.g. peak at 8:00)
            morning_peak = 0.8 * np.exp(-0.5 * ((hour - 8.0) / 1.0) ** 2)
            # Evening peak (e.g. peak at 19:00)
            evening_peak = 1.5 * np.exp(-0.5 * ((hour - 19.0) / 1.5) ** 2)

            load += morning_peak + evening_peak

            # HVAC adjustment based on temp (heating below 15°C, cooling above 25°C)
            if temp < 15.0:
                load += 0.08 * (15.0 - temp)  # Heating load
            elif temp > 25.0:
                load += 0.12 * (temp - 25.0)  # Cooling load

            # Weekend shift: overall consumption slightly higher and shifted later
            if is_weekend:
                load *= 1.15

            consumption.append(max(0.05, load))

        return np.array(consumption)

    def update_battery_state(
        self,
        battery_energy: float,
        solar: float,
        consumption: float,
        dt: float = 5.0 / 60.0,
    ) -> tuple[float, float, float, float]:
        """Update battery state of charge and return updated energy and flows.

        Returns:
            Tuple of (new_battery_energy, net_flow_kw, wasted_solar_kw, grid_import_kw)
        """
        diff = solar - consumption
        wasted_solar = 0.0
        grid_import = 0.0
        prev_energy = battery_energy

        if diff > 0:  # Excess Solar: charge battery
            charge_power = diff * self.charge_efficiency
            new_energy = battery_energy + charge_power * dt
            max_energy = self.battery_capacity_kwh

            if new_energy > max_energy:
                # Battery is full, excess solar is wasted (or exported)
                wasted_power = (new_energy - max_energy) / dt / self.charge_efficiency
                wasted_solar = wasted_power
                new_energy = max_energy
                net_flow = (max_energy - prev_energy) / dt
            else:
                new_energy = new_energy
                net_flow = diff
        else:  # Deficit: discharge battery
            discharge_power = abs(diff) / self.discharge_efficiency
            new_energy = battery_energy - discharge_power * dt
            min_energy = (self.min_soc / 100.0) * self.battery_capacity_kwh

            if new_energy < min_energy:
                # Battery empty, import remaining deficit from grid
                shortage_power = (min_energy - new_energy) / dt * self.discharge_efficiency
                grid_import = shortage_power
                new_energy = min_energy
                net_flow = (min_energy - prev_energy) / dt
            else:
                new_energy = new_energy
                net_flow = diff

        return new_energy, net_flow, wasted_solar, grid_import

    def run(self, start_date: str, days: int, initial_soc: float = 50.0) -> pd.DataFrame:
        """Generate a complete simulated history DataFrame."""
        timestamps = pd.date_range(
            start=start_date, periods=days * 288, freq="5min", name="timestamp"
        )
        n_steps = len(timestamps)

        # Generate smooth temperature variations (daily cycles + seasonal trends)
        # Seasonal trend: peaks in summer (June 21, day 172)
        base_temp = []
        for ts in timestamps:
            day_of_year = ts.dayofyear
            hour = ts.hour + ts.minute / 60.0
            # Seasonal baseline
            season_base = 15.0 + 10.0 * np.cos(2.0 * np.pi * (day_of_year - 172) / 365.25)
            # Daily fluctuation peaking at 15:00
            daily_var = 5.0 * np.cos(2.0 * np.pi * (hour - 15.0) / 24.0)
            base_temp.append(season_base + daily_var)

        temperatures = np.array(base_temp) + self.rng.normal(0, 1.0, n_steps)

        # Cloud cover model (random walk between 0 and 1, smoothed)
        cloud_cover = np.zeros(n_steps)
        current_cloud = 0.3
        for i in range(n_steps):
            current_cloud = np.clip(current_cloud + self.rng.normal(0, 0.05), 0.0, 1.0)
            cloud_cover[i] = current_cloud

        # Run Solar and Consumption
        solar = self.simulate_solar(timestamps, cloud_cover)
        consumption = self.simulate_consumption(timestamps, temperatures)

        # Physics simulation for battery charging/discharging
        soc = np.zeros(n_steps)
        current_soc = initial_soc
        battery_energy = (current_soc / 100.0) * self.battery_capacity_kwh
        net_flow = np.zeros(n_steps)
        wasted_solar = np.zeros(n_steps)
        grid_import = np.zeros(n_steps)

        # 5 minutes in hours
        dt = 5.0 / 60.0

        for i in range(n_steps):
            battery_energy, net_flow_val, wasted, imported = self.update_battery_state(
                battery_energy, solar[i], consumption[i], dt
            )
            wasted_solar[i] = wasted
            grid_import[i] = imported
            net_flow[i] = net_flow_val
            current_soc = (battery_energy / self.battery_capacity_kwh) * 100.0
            soc[i] = current_soc

        df = pd.DataFrame(
            {
                "solar_production": solar,
                "consumption": consumption,
                "soc": soc,
                "temperature": temperatures,
                "cloud_cover": cloud_cover,
                "wasted_solar": wasted_solar,
                "grid_import": grid_import,
                "net_flow": net_flow,
            },
            index=timestamps,
        )
        return df
