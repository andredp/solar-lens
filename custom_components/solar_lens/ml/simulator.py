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
        max_charge_power_kw: float = float("inf"),
        max_discharge_power_kw: float = float("inf"),
    ) -> tuple[float, float, float, float]:
        """Update battery state of charge and return updated energy and flows.

        Args:
            battery_energy: Current energy in the battery (kWh).
            solar: Solar production power (kW) during this step.
            consumption: House consumption power (kW) during this step.
            dt: Time step duration (hours).
            max_charge_power_kw: Maximum DC charge power limit (kW).
            max_discharge_power_kw: Maximum DC discharge power limit (kW).

        Returns:
            Tuple of (new_battery_energy, net_flow_kw, wasted_solar_kw, grid_import_kw)
            where net_flow_kw is the DC terminal power flow
            (positive for charge, negative for discharge).
        """
        diff = solar - consumption
        wasted_solar = 0.0
        grid_import = 0.0

        if diff > 0:  # Excess Solar: charge battery
            potential_dc_charge = diff * self.charge_efficiency
            dc_charge = min(potential_dc_charge, max_charge_power_kw)
            new_energy = battery_energy + dc_charge * dt
            max_energy = self.battery_capacity_kwh

            if new_energy > max_energy:
                # Battery is full
                new_energy = max_energy
                actual_dc_charge = (max_energy - battery_energy) / dt
                wasted_solar = diff - (actual_dc_charge / self.charge_efficiency)
                net_flow = actual_dc_charge
            else:
                wasted_solar = diff - (dc_charge / self.charge_efficiency)
                net_flow = dc_charge
        else:  # Deficit or zero: discharge battery
            potential_dc_discharge = abs(diff) / self.discharge_efficiency
            dc_discharge = min(potential_dc_discharge, max_discharge_power_kw)
            new_energy = battery_energy - dc_discharge * dt
            min_energy = (self.min_soc / 100.0) * self.battery_capacity_kwh

            if new_energy < min_energy:
                # Battery empty
                new_energy = min_energy
                actual_dc_discharge = (battery_energy - min_energy) / dt
                grid_import = abs(diff) - (actual_dc_discharge * self.discharge_efficiency)
                net_flow = -actual_dc_discharge
            else:
                grid_import = abs(diff) - (dc_discharge * self.discharge_efficiency)
                net_flow = -dc_discharge

        # Float correction to prevent tiny negative wastes/imports
        return (
            new_energy,
            float(net_flow),
            max(0.0, float(wasted_solar)),
            max(0.0, float(grid_import)),
        )

    def run_simulation(
        self,
        initial_soc: float,
        solar_forecast: list[float] | np.ndarray,
        consumption_forecast: list[float] | np.ndarray,
        max_charge_power_forecast: list[float] | np.ndarray | None = None,
        max_discharge_power_forecast: list[float] | np.ndarray | None = None,
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

        if max_charge_power_forecast is None:
            max_charge_power_forecast = [float("inf")] * n_steps
        if max_discharge_power_forecast is None:
            max_discharge_power_forecast = [float("inf")] * n_steps

        current_soc = initial_soc
        battery_energy = (current_soc / 100.0) * self.battery_capacity_kwh

        for i in range(n_steps):
            battery_energy, net_flow_val, wasted, imported = self.update_battery_state(
                battery_energy,
                solar_forecast[i],
                consumption_forecast[i],
                dt,
                max_charge_power_forecast[i],
                max_discharge_power_forecast[i],
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
