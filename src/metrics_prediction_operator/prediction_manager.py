"""Prediction manager for handling metric predictions."""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime

import pandas as pd
from prophet import Prophet

from .config import MetricConfig
from .model import generate_predictions, train_model
from .prometheus import fetch_prometheus_data, process_dataframe_for_prophet

logger = logging.getLogger(__name__)


@dataclass
class MetricPrediction:
    """Stores prediction data for a single metric."""

    metric_name: str
    query: str
    model: Prophet | None = None
    data_df: pd.DataFrame = field(default_factory=lambda: pd.DataFrame(columns=["ds", "y"]))
    predictions_df: pd.DataFrame = field(
        default_factory=lambda: pd.DataFrame(columns=["ds", "yhat", "yhat_lower", "yhat_upper"])
    )
    last_updated: datetime | None = None
    prediction_horizon_hours: int = 24
    is_training: bool = False
    error_message: str | None = None


class PredictionManager:
    """Manages predictions for multiple metrics."""

    def __init__(
        self,
        prometheus_url: str,
        metrics: list[MetricConfig],
        update_interval_minutes: int = 30,
        hpo_enabled: bool = True,
        hpo_n_trials: int = 50,
    ) -> None:
        """Initialize the prediction manager.

        Args:
            prometheus_url: URL of the Prometheus server.
            metrics: List of metric configurations.
            update_interval_minutes: Interval between prediction updates.
            hpo_enabled: Whether to use HPO for parameter tuning.
            hpo_n_trials: Number of trials for HPO.
        """
        self.prometheus_url = prometheus_url
        self.metrics = metrics
        self.update_interval_minutes = update_interval_minutes
        self.hpo_enabled = hpo_enabled
        self.hpo_n_trials = hpo_n_trials
        self._predictions: dict[str, MetricPrediction] = {}
        self._running = False
        self._update_task: asyncio.Task[None] | None = None

        # Initialize prediction objects for each metric
        for metric in metrics:
            self._predictions[metric.name] = MetricPrediction(
                metric_name=metric.name,
                query=metric.query,
                prediction_horizon_hours=metric.prediction_horizon_hours,
            )

    @property
    def predictions(self) -> dict[str, MetricPrediction]:
        """Get all metric predictions."""
        return self._predictions

    async def update_metric(self, metric_name: str) -> None:
        """Update predictions for a single metric.

        Args:
            metric_name: Name of the metric to update.
        """
        if metric_name not in self._predictions:
            logger.error(f"Unknown metric: {metric_name}")
            return

        prediction = self._predictions[metric_name]

        if prediction.is_training:
            logger.warning(f"Metric {metric_name} is already being updated")
            return

        prediction.is_training = True
        prediction.error_message = None

        try:
            # Fetch data from Prometheus
            logger.info(f"Fetching data for metric: {metric_name}")
            df = await fetch_prometheus_data(
                self.prometheus_url,
                prediction.query,
            )

            if df.empty:
                prediction.error_message = "No data returned from Prometheus"
                logger.warning(f"No data for metric: {metric_name}")
                return

            # Process the data
            df = process_dataframe_for_prophet(df)
            prediction.data_df = df

            # Train model (runs in thread pool to avoid blocking)
            logger.info(f"Training model for metric: {metric_name}")
            loop = asyncio.get_event_loop()
            model = await loop.run_in_executor(
                None,
                lambda: train_model(
                    df,
                    use_hpo=self.hpo_enabled,
                    n_trials=self.hpo_n_trials,
                ),
            )
            prediction.model = model

            # Generate predictions
            predictions_df = await loop.run_in_executor(
                None,
                lambda: generate_predictions(
                    model,
                    periods=prediction.prediction_horizon_hours,
                    freq="h",
                ),
            )
            prediction.predictions_df = predictions_df
            prediction.last_updated = datetime.now(tz=UTC)

            logger.info(f"Successfully updated predictions for: {metric_name}")

        except Exception as e:
            prediction.error_message = str(e)
            logger.exception(f"Error updating metric {metric_name}: {e}")

        finally:
            prediction.is_training = False

    async def update_all_metrics(self) -> None:
        """Update predictions for all configured metrics."""
        logger.info("Updating all metric predictions")
        tasks = [self.update_metric(metric.name) for metric in self.metrics]
        await asyncio.gather(*tasks, return_exceptions=True)
        logger.info("Finished updating all metrics")

    async def _update_loop(self) -> None:
        """Background loop to periodically update predictions."""
        while self._running:
            await self.update_all_metrics()
            await asyncio.sleep(self.update_interval_minutes * 60)

    async def start(self) -> None:
        """Start the background update loop."""
        if self._running:
            return

        self._running = True
        self._update_task = asyncio.create_task(self._update_loop())
        logger.info("Started prediction manager")

    async def stop(self) -> None:
        """Stop the background update loop."""
        self._running = False
        if self._update_task:
            self._update_task.cancel()
            try:
                await self._update_task
            except asyncio.CancelledError:
                pass
            self._update_task = None
        logger.info("Stopped prediction manager")

    def get_prediction(self, metric_name: str) -> MetricPrediction | None:
        """Get prediction for a specific metric.

        Args:
            metric_name: Name of the metric.

        Returns:
            MetricPrediction object or None if not found.
        """
        return self._predictions.get(metric_name)

    def get_current_prediction_value(self, metric_name: str) -> float | None:
        """Get the current predicted value for a metric.

        Args:
            metric_name: Name of the metric.

        Returns:
            Current predicted value or None if not available.
        """
        prediction = self.get_prediction(metric_name)
        if prediction is None or prediction.predictions_df.empty:
            return None

        # Prophet returns timezone-naive timestamps
        now = datetime.now()
        df = prediction.predictions_df.copy()

        # Find the closest prediction to now
        df["time_diff"] = abs((df["ds"] - now).dt.total_seconds())
        closest_idx = df["time_diff"].idxmin()
        return float(df.loc[closest_idx, "yhat"])
