"""Prophet model with Optuna hyperparameter optimization."""

import logging
from datetime import datetime, timedelta
from typing import Any

import optuna
import pandas as pd
from prophet import Prophet

logger = logging.getLogger(__name__)


# Default Prophet parameters
DEFAULT_PARAMS: dict[str, Any] = {
    "changepoint_prior_scale": 0.05,
    "seasonality_prior_scale": 10.0,
    "holidays_prior_scale": 10.0,
    "seasonality_mode": "additive",
    "changepoint_range": 0.8,
    "yearly_seasonality": "auto",
    "weekly_seasonality": "auto",
    "daily_seasonality": "auto",
}


def create_prophet_model(params: dict[str, Any] | None = None) -> Prophet:
    """Create a Prophet model with the given parameters.

    Args:
        params: Dictionary of Prophet parameters. Uses defaults if not provided.

    Returns:
        Configured Prophet model instance.
    """
    model_params = {**DEFAULT_PARAMS, **(params or {})}
    return Prophet(**model_params)


def objective(
    trial: optuna.Trial,
    df: pd.DataFrame,
    validation_period_days: int = 7,
) -> float:
    """Optuna objective function for Prophet hyperparameter optimization.

    Args:
        trial: Optuna trial object.
        df: DataFrame with 'ds' and 'y' columns.
        validation_period_days: Number of days to use for validation.

    Returns:
        Mean absolute error on the validation set.
    """
    # Define hyperparameter search space
    params = {
        "changepoint_prior_scale": trial.suggest_float(
            "changepoint_prior_scale", 0.001, 0.5, log=True
        ),
        "seasonality_prior_scale": trial.suggest_float(
            "seasonality_prior_scale", 0.01, 10.0, log=True
        ),
        "holidays_prior_scale": trial.suggest_float(
            "holidays_prior_scale", 0.01, 10.0, log=True
        ),
        "seasonality_mode": trial.suggest_categorical(
            "seasonality_mode", ["additive", "multiplicative"]
        ),
        "changepoint_range": trial.suggest_float("changepoint_range", 0.7, 0.95),
    }

    # Split data into train and validation sets
    cutoff = df["ds"].max() - timedelta(days=validation_period_days)
    train_df = df[df["ds"] <= cutoff].copy()
    valid_df = df[df["ds"] > cutoff].copy()

    if len(train_df) < 2 or len(valid_df) < 1:
        # Not enough data for validation
        return float("inf")

    # Suppress Prophet logs during optimization
    model = Prophet(
        **params,
        yearly_seasonality="auto",
        weekly_seasonality="auto",
        daily_seasonality="auto",
    )
    model.fit(train_df)

    # Make predictions on validation set
    forecast = model.predict(valid_df[["ds"]])
    mae = (abs(valid_df["y"].values - forecast["yhat"].values)).mean()

    return float(mae)


def optimize_prophet_params(
    df: pd.DataFrame,
    n_trials: int = 50,
    validation_period_days: int = 7,
) -> dict[str, Any]:
    """Optimize Prophet hyperparameters using Optuna.

    Args:
        df: DataFrame with 'ds' and 'y' columns.
        n_trials: Number of optimization trials.
        validation_period_days: Number of days to use for validation.

    Returns:
        Dictionary of optimized parameters.
    """
    if len(df) < 14:  # Minimum data requirement
        logger.warning("Not enough data for HPO, using default parameters")
        return DEFAULT_PARAMS.copy()

    # Suppress Optuna and Prophet logging during optimization
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    study = optuna.create_study(direction="minimize")
    study.optimize(
        lambda trial: objective(trial, df, validation_period_days),
        n_trials=n_trials,
        show_progress_bar=False,
    )

    # Merge optimized params with defaults
    optimized_params = {**DEFAULT_PARAMS, **study.best_params}
    logger.info(f"Optimized parameters: {optimized_params}")

    return optimized_params


def train_model(
    df: pd.DataFrame,
    use_hpo: bool = True,
    n_trials: int = 50,
) -> Prophet:
    """Train a Prophet model with optional hyperparameter optimization.

    Args:
        df: DataFrame with 'ds' and 'y' columns.
        use_hpo: Whether to use Optuna for hyperparameter optimization.
        n_trials: Number of Optuna trials if HPO is enabled.

    Returns:
        Trained Prophet model.
    """
    if use_hpo and len(df) >= 14:
        params = optimize_prophet_params(df, n_trials=n_trials)
    else:
        params = DEFAULT_PARAMS.copy()

    model = create_prophet_model(params)
    model.fit(df)
    return model


def generate_predictions(
    model: Prophet,
    periods: int,
    freq: str = "h",
) -> pd.DataFrame:
    """Generate predictions using a trained Prophet model.

    Args:
        model: Trained Prophet model.
        periods: Number of periods to predict.
        freq: Frequency of predictions ('h' for hourly, 'D' for daily).

    Returns:
        DataFrame with predictions including 'ds', 'yhat', 'yhat_lower', 'yhat_upper'.
    """
    future = model.make_future_dataframe(periods=periods, freq=freq)
    forecast = model.predict(future)
    return forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]]


def create_sample_dataframe() -> pd.DataFrame:
    """Create a sample DataFrame for testing/demo purposes.

    Returns:
        DataFrame with 'ds' and 'y' columns containing sample data.
    """
    now = datetime.now()  # Prophet requires timezone-naive timestamps
    dates = pd.date_range(end=now, periods=168, freq="h")  # 7 days of hourly data

    # Create synthetic seasonal pattern
    import numpy as np

    np.random.seed(42)
    y = (
        100
        + 10 * np.sin(np.arange(len(dates)) * 2 * np.pi / 24)  # Daily pattern
        + 5 * np.sin(np.arange(len(dates)) * 2 * np.pi / 168)  # Weekly pattern
        + np.random.normal(0, 2, len(dates))  # Noise
    )

    return pd.DataFrame({"ds": dates, "y": y})
