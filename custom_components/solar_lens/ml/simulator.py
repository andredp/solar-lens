"""Physical battery simulator for Solar Lens battery prediction."""

import numpy as np


class SolarBatterySimulator:
    """Simulates physical battery charging, discharging, limits, and efficiency."""

    def __init__(
        self,
        battery_capacity_kwh: float,
        charge_efficiency: float = 0.95,
        discharge_efficiency: float = 0.95,
        min_soc: float = 10.0,
    ) -> None:
        """Initialize the simulator with physical configuration."""
        self.battery_capacity_kwh = battery_capacity_kwh
        self.charge_efficiency = charge_efficiency
        self.discharge_efficiency = discharge_efficiency
        self.min_soc = min_soc

    def update_battery_state(
        self,
        battery_energy: float,
        solar: float,
        consumption: float,
        dt: float,
    ) -> tuple[float, float, float, float]:
        """Update battery state of charge and return updated energy and flows.

        Args:
            battery_energy: Current energy in the battery (kWh).
            solar: Solar production power (kW) during this step.
            consumption: House consumption power (kW) during this step.
            dt: Time step duration (hours).

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

    def run_simulation(
        self,
        initial_soc: float,
        solar_forecast: list[float] | np.ndarray,
        consumption_forecast: list[float] | np.ndarray,
        dt: float = 5.0 / 60.0,
    ) -> dict[str, list[float]]:
        """Run the simulation step-by-step for the given forecasts.

        Returns:
            Dictionary containing curves for:
            - "soc": list of SoC values (%)
            - "wasted_solar": list of wasted solar power (kW)
            - "grid_import": list of grid import power (kW)
            - "net_flow": list of battery net power flow (kW)
        """
        n_steps = len(solar_forecast)
        soc = []
        wasted_solar = []
        grid_import = []
        net_flow = []

        current_soc = initial_soc
        battery_energy = (current_soc / 100.0) * self.battery_capacity_kwh

        for i in range(n_steps):
            battery_energy, net_flow_val, wasted, imported = self.update_battery_state(
                battery_energy, solar_forecast[i], consumption_forecast[i], dt
            )
            wasted_solar.append(float(wasted))
            grid_import.append(float(imported))
            net_flow.append(float(net_flow_val))
            current_soc = (battery_energy / self.battery_capacity_kwh) * 100.0
            soc.append(float(current_soc))

        return {
            "soc": soc,
            "wasted_solar": wasted_solar,
            "grid_import": grid_import,
            "net_flow": net_flow,
        }
