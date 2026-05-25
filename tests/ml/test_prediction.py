"""Unit tests verifying predictability of simulated data using HistGradientBoostingRegressor."""

import sys
from pathlib import Path

import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import root_mean_squared_error

# Add custom_components to path so we can import solar_lens
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "custom_components"))

from solar_lens.ml.feature_engineering import compute_rolling_mean, extract_time_features
from tests.ml.simulator import SolarBatterySimulator


def test_battery_soc_predictability() -> None:
    """Train a HistGradientBoostingRegressor on simulated data to verify future SoC prediction.

    Ensures that the model's RMSE is less than 8.0% SoC error.
    """
    # 1. Generate 30 days of training data and 5 days of testing data
    sim = SolarBatterySimulator(battery_capacity_kwh=7.2, seed=42)
    df_train = sim.run("2026-06-01", days=30, initial_soc=50.0)
    df_test = sim.run("2026-07-01", days=5, initial_soc=50.0)

    # 2. Apply feature engineering to both sets
    # We want to predict the next step's SoC (5 minutes in the future) or future 1 hour SoC
    # Let's predict SoC 1 hour in the future (12 steps of 5 mins)
    forecast_steps = 12

    # Feature engineering pipeline
    def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
        processed = df.copy()
        # Extract time features
        time_feats = extract_time_features(processed.index)
        for k, v in time_feats.items():
            processed[k] = v

        # Add rolling averages for consumption and solar
        for col in ["consumption", "solar_production"]:
            for window in [3, 12]:
                processed[f"{col}_roll_mean_{window}st"] = compute_rolling_mean(
                    processed[col].to_numpy(), window
                )

        # Define target: SoC in 1 hour
        processed["target_soc"] = processed["soc"].shift(-forecast_steps)

        # Drop NaN rows due to rolling and shift
        processed = processed.dropna()
        return processed

    train_data = prepare_data(df_train)
    test_data = prepare_data(df_test)

    # Define feature columns
    features = [
        "soc",
        "temperature",
        "cloud_cover",
        "time_sin",
        "time_cos",
        "day_sin",
        "day_cos",
        "is_weekend",
        "consumption_roll_mean_3st",
        "consumption_roll_mean_12st",
        "solar_production_roll_mean_3st",
        "solar_production_roll_mean_12st",
    ]

    X_train = train_data[features]
    y_train = train_data["target_soc"]
    X_test = test_data[features]
    y_test = test_data["target_soc"]

    # 3. Train HistGradientBoostingRegressor
    model = HistGradientBoostingRegressor(random_state=42, max_iter=50)
    model.fit(X_train, y_train)

    # 4. Predict and evaluate
    predictions = model.predict(X_test)
    rmse = root_mean_squared_error(y_test, predictions)

    # Since the simulated system is highly deterministic and physical,
    # the regressor should be able to predict the SoC 1 hour in the future with high accuracy.
    # RMSE should be low (e.g. less than 8.0% SoC error).
    assert rmse < 8.0
