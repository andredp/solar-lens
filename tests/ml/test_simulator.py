"""Unit tests for the physical battery and solar simulator."""

import sys
from pathlib import Path

import pytest

# Add custom_components to path so we can import solar_lens
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "custom_components"))

from tests.ml.simulator import SolarBatterySimulator


def test_simulator_dataframe_structure() -> None:
    """Verify that the simulator outputs the correct shape and column types."""
    sim = SolarBatterySimulator(battery_capacity_kwh=10.0, seed=42)
    days = 3
    df = sim.run("2026-06-01", days=days, initial_soc=50.0)

    # 3 days * 288 periods/day = 864 periods
    assert len(df) == days * 288
    expected_cols = {
        "solar_production",
        "consumption",
        "soc",
        "temperature",
        "cloud_cover",
        "wasted_solar",
        "grid_import",
        "net_flow",
    }
    assert expected_cols.issubset(df.columns)


def test_simulator_physical_limits() -> None:
    """Verify that basic physics and operational constraints are respected by the simulator."""
    min_soc = 15.0
    sim = SolarBatterySimulator(battery_capacity_kwh=5.0, min_soc=min_soc, seed=42)
    df = sim.run("2026-01-01", days=5, initial_soc=20.0)

    # Solar is non-negative
    assert (df["solar_production"] >= 0).all()

    # Consumption is non-negative
    assert (df["consumption"] >= 0).all()

    # SoC is strictly between min_soc and 100%
    assert (df["soc"] >= min_soc - 0.01).all()
    assert (df["soc"] <= 100).all()

    # Wasted solar is only present when battery is full (SoC > 99.9%)
    full_battery_indices = df["soc"] > 99.9
    wasted_solar_outside_full = df.loc[~full_battery_indices, "wasted_solar"]
    assert pytest.approx(wasted_solar_outside_full.max()) == 0.0

    # Grid import is only present when battery is empty (SoC < min_soc + 0.1%)
    empty_battery_indices = df["soc"] < min_soc + 0.1
    grid_import_outside_empty = df.loc[~empty_battery_indices, "grid_import"]
    assert pytest.approx(grid_import_outside_empty.max()) == 0.0


def test_battery_energy_conservation_analytical() -> None:
    """Verify energy conservation under specific analytical scenarios."""
    sim = SolarBatterySimulator(
        battery_capacity_kwh=10.0,
        charge_efficiency=0.90,
        discharge_efficiency=0.90,
        min_soc=10.0,
    )
    dt = 1.0  # 1 hour

    # 1. Charging scenario (excess solar)
    # Start at 5.0 kWh (50% SoC). Solar = 2.0 kW, Consumption = 0.0 kW.
    # Expected stored energy = 2.0 kW * 0.90 * 1.0 h = 1.8 kWh.
    # New energy should be 5.0 + 1.8 = 6.8 kWh.
    new_energy, net_flow, wasted, imported = sim.update_battery_state(
        battery_energy=5.0, solar=2.0, consumption=0.0, dt=dt
    )
    assert pytest.approx(new_energy) == 6.8
    assert pytest.approx(net_flow) == 2.0
    assert pytest.approx(wasted) == 0.0
    assert pytest.approx(imported) == 0.0

    # 2. Charging scenario with overflow (wasted solar)
    # Start at 9.0 kWh. Solar = 2.0 kW, Consumption = 0.0 kW.
    # Max charge capacity is 1.0 kWh.
    # Required input energy to store 1.0 kWh is 1.0 / 0.90 = 1.1111... kWh
    # (approx 1.1111 kW for 1h).
    # Remaining 2.0 - 1.1111... = 0.8888... kW is wasted.
    new_energy, net_flow, wasted, imported = sim.update_battery_state(
        battery_energy=9.0, solar=2.0, consumption=0.0, dt=dt
    )
    assert pytest.approx(new_energy) == 10.0
    assert pytest.approx(wasted) == 2.0 - (1.0 / 0.90)
    assert pytest.approx(imported) == 0.0

    # 3. Discharging scenario (deficit)
    # Start at 5.0 kWh (50% SoC). Solar = 0.0 kW, Consumption = 1.8 kW.
    # Expected battery energy reduction = 1.8 kW / 0.90 * 1.0 h = 2.0 kWh.
    # New energy should be 5.0 - 2.0 = 3.0 kWh.
    new_energy, net_flow, wasted, imported = sim.update_battery_state(
        battery_energy=5.0, solar=0.0, consumption=1.8, dt=dt
    )
    assert pytest.approx(new_energy) == 3.0
    assert pytest.approx(net_flow) == -1.8
    assert pytest.approx(wasted) == 0.0
    assert pytest.approx(imported) == 0.0

    # 4. Discharging scenario with underflow (grid import)
    # Start at 2.0 kWh (20% SoC). Min energy is 1.0 kWh (10% SoC).
    # Solar = 0.0 kW, Consumption = 1.8 kW.
    # Discharging 1.8 kW for 1h requires 1.8 / 0.90 = 2.0 kWh.
    # But battery can only discharge 1.0 kWh (takes 1.0 * 0.90 = 0.9 kWh load).
    # Deficit load is 1.8 - 0.9 = 0.9 kW. This is imported from the grid.
    new_energy, net_flow, wasted, imported = sim.update_battery_state(
        battery_energy=2.0, solar=0.0, consumption=1.8, dt=dt
    )
    assert pytest.approx(new_energy) == 1.0
    assert pytest.approx(imported) == 1.8 - (1.0 * 0.90)
    assert pytest.approx(wasted) == 0.0
