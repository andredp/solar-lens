"""Unit tests for the production physical battery simulator."""

import sys
from pathlib import Path

import pytest

# Add custom_components to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "custom_components"))

from solar_lens.ml.simulator import SolarBatterySimulator


def test_production_simulator_charging_no_overflow() -> None:
    """Test battery charging when there is no overflow and no limits."""
    sim = SolarBatterySimulator(battery_capacity_kwh=10.0, charge_efficiency=0.90)

    # 5.0 kWh current energy. Solar = 2.0 kW, Consumption = 1.0 kW -> diff = 1.0 kW.
    # DC charge power entering battery = 1.0 * 0.90 = 0.90 kW.
    # In 1 hour (dt=1.0), battery energy increases by 0.90 kWh.
    new_energy, net_flow, wasted, imported = sim.update_battery_state(
        battery_energy=5.0, solar=2.0, consumption=1.0, dt=1.0
    )

    assert new_energy == 5.9
    assert net_flow == 0.9  # DC terminal flow
    assert wasted == 0.0
    assert imported == 0.0


def test_production_simulator_charging_overflow() -> None:
    """Test battery charging when battery capacity is exceeded."""
    sim = SolarBatterySimulator(battery_capacity_kwh=10.0, charge_efficiency=0.90)

    # 9.5 kWh current energy. Solar = 2.0 kW, Consumption = 0.0 kW -> diff = 2.0 kW.
    # Remaining capacity is 0.5 kWh.
    # Charging to full takes 0.5 / 0.90 = 0.5555... kW AC power.
    # Wasted solar power = 2.0 - 0.5555... = 1.4444... kW.
    new_energy, net_flow, wasted, imported = sim.update_battery_state(
        battery_energy=9.5, solar=2.0, consumption=0.0, dt=1.0
    )

    assert new_energy == 10.0
    assert pytest.approx(net_flow) == 0.5
    assert pytest.approx(wasted) == 2.0 - (0.5 / 0.90)
    assert imported == 0.0


def test_production_simulator_charging_capped_by_bms() -> None:
    """Test battery charging when capped by BMS charge limit."""
    sim = SolarBatterySimulator(battery_capacity_kwh=10.0, charge_efficiency=0.90)

    # 5.0 kWh current energy. Solar = 5.0 kW, Consumption = 0.0 kW -> diff = 5.0 kW.
    # DC limit is 1.8 kW (e.g. cold battery).
    # potential DC charge is 5.0 * 0.90 = 4.5 kW. Capped to 1.8 kW DC.
    # AC power actually used to charge = 1.8 / 0.90 = 2.0 kW.
    # Wasted solar = 5.0 - 2.0 = 3.0 kW.
    new_energy, net_flow, wasted, imported = sim.update_battery_state(
        battery_energy=5.0,
        solar=5.0,
        consumption=0.0,
        dt=1.0,
        max_charge_power_kw=1.8,
    )

    assert new_energy == 6.8
    assert net_flow == 1.8
    assert pytest.approx(wasted) == 3.0
    assert imported == 0.0


def test_production_simulator_discharging_no_underflow() -> None:
    """Test battery discharging under normal conditions."""
    sim = SolarBatterySimulator(battery_capacity_kwh=10.0, discharge_efficiency=0.90, min_soc=10.0)

    # 5.0 kWh current energy. Solar = 0.0 kW, Consumption = 1.8 kW -> diff = -1.8 kW.
    # DC discharge power drawn = 1.8 / 0.90 = 2.0 kW.
    # In 1 hour (dt=1.0), energy decreases by 2.0 kWh.
    new_energy, net_flow, wasted, imported = sim.update_battery_state(
        battery_energy=5.0, solar=0.0, consumption=1.8, dt=1.0
    )

    assert new_energy == 3.0
    assert net_flow == -2.0  # DC terminal flow (negative for discharge)
    assert wasted == 0.0
    assert imported == 0.0


def test_production_simulator_discharging_underflow() -> None:
    """Test battery discharging when SoC hits the floor."""
    sim = SolarBatterySimulator(battery_capacity_kwh=10.0, discharge_efficiency=0.90, min_soc=10.0)
    min_energy = 1.0  # 10% of 10.0

    # 2.0 kWh current energy. Solar = 0.0 kW, Consumption = 1.8 kW.
    # Remaining usable energy is 1.0 kWh.
    # Drawing 1.0 kWh DC covers 1.0 * 0.90 = 0.90 kW AC load.
    # Remaining deficit = 1.8 - 0.90 = 0.90 kW imported from grid.
    new_energy, net_flow, wasted, imported = sim.update_battery_state(
        battery_energy=2.0, solar=0.0, consumption=1.8, dt=1.0
    )

    assert new_energy == min_energy
    assert net_flow == -1.0
    assert wasted == 0.0
    assert pytest.approx(imported) == 0.90


def test_production_simulator_discharging_capped_by_bms() -> None:
    """Test battery discharging when capped by BMS discharge limit."""
    sim = SolarBatterySimulator(battery_capacity_kwh=10.0, discharge_efficiency=0.90, min_soc=10.0)

    # 5.0 kWh current energy. Solar = 0.0 kW, Consumption = 4.5 kW.
    # BMS limit is 1.8 kW DC.
    # Potential DC discharge needed is 4.5 / 0.90 = 5.0 kW DC. Capped to 1.8 kW DC.
    # AC power covered by battery = 1.8 * 0.90 = 1.62 kW AC.
    # Remaining load imported = 4.5 - 1.62 = 2.88 kW.
    new_energy, net_flow, wasted, imported = sim.update_battery_state(
        battery_energy=5.0,
        solar=0.0,
        consumption=4.5,
        dt=1.0,
        max_discharge_power_kw=1.8,
    )

    assert new_energy == 3.2
    assert net_flow == -1.8
    assert wasted == 0.0
    assert pytest.approx(imported) == 2.88


def test_production_simulator_run_simulation() -> None:
    """Test running a sequential prediction simulation."""
    sim = SolarBatterySimulator(
        battery_capacity_kwh=10.0, charge_efficiency=0.90, discharge_efficiency=0.90
    )

    # Run 3-step simulation
    # Step 1: Solar = 2.0 kW, Consumption = 0.0 kW (Charge)
    # Step 2: Solar = 0.0 kW, Consumption = 1.8 kW (Discharge)
    # Step 3: Solar = 0.0 kW, Consumption = 0.0 kW (Idle)
    solar_forecast = [2.0, 0.0, 0.0]
    consumption_forecast = [0.0, 1.8, 0.0]

    res = sim.run_simulation(
        initial_soc=50.0,
        solar_forecast=solar_forecast,
        consumption_forecast=consumption_forecast,
        dt=1.0,
    )

    # Step 1: 5.0 kWh -> +1.8 kWh -> 6.8 kWh (68% SoC). net_flow = 1.8
    # Step 2: 6.8 kWh -> -2.0 kWh -> 4.8 kWh (48% SoC). net_flow = -2.0
    # Step 3: 4.8 kWh -> idle -> 4.8 kWh (48% SoC). net_flow = 0.0
    assert res["soc"] == [68.0, 48.0, 48.0]
    assert res["net_flow"] == [1.8, -2.0, 0.0]
    assert res["wasted_solar"] == [0.0, 0.0, 0.0]
    assert res["grid_import"] == [0.0, 0.0, 0.0]
