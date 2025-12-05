"""FastAPI application for serving prediction metrics."""

import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Response
from prometheus_client import CONTENT_TYPE_LATEST, Gauge, generate_latest

from .config import AppConfig
from .prediction_manager import PredictionManager

logger = logging.getLogger(__name__)

# Global prediction manager
prediction_manager: PredictionManager | None = None

# Prometheus gauges for predictions
prediction_gauges: dict[str, Gauge] = {}
prediction_lower_gauges: dict[str, Gauge] = {}
prediction_upper_gauges: dict[str, Gauge] = {}
last_updated_gauge: Gauge | None = None


def setup_prometheus_metrics(config: AppConfig) -> None:
    """Set up Prometheus metrics gauges for all configured metrics.

    Args:
        config: Application configuration.
    """
    global last_updated_gauge

    for metric_config in config.metrics:
        metric_name = metric_config.name
        safe_name = metric_name.replace("-", "_").replace(".", "_")

        prediction_gauges[metric_name] = Gauge(
            f"{safe_name}_prediction",
            f"Predicted value for {metric_name}",
        )

        prediction_lower_gauges[metric_name] = Gauge(
            f"{safe_name}_prediction_lower",
            f"Lower bound of prediction for {metric_name}",
        )

        prediction_upper_gauges[metric_name] = Gauge(
            f"{safe_name}_prediction_upper",
            f"Upper bound of prediction for {metric_name}",
        )

    last_updated_gauge = Gauge(
        "prediction_last_updated_timestamp",
        "Timestamp of last prediction update",
        ["metric_name"],
    )


def update_prometheus_metrics() -> None:
    """Update Prometheus metrics with current prediction values."""
    global prediction_manager, last_updated_gauge

    if prediction_manager is None:
        return

    for metric_name, prediction in prediction_manager.predictions.items():
        if metric_name not in prediction_gauges:
            continue

        if prediction.predictions_df.empty:
            continue

        # Get current prediction value
        current_value = prediction_manager.get_current_prediction_value(metric_name)
        if current_value is not None:
            prediction_gauges[metric_name].set(current_value)

        # Get bounds from latest prediction
        if not prediction.predictions_df.empty:
            latest = prediction.predictions_df.iloc[-1]
            prediction_lower_gauges[metric_name].set(float(latest["yhat_lower"]))
            prediction_upper_gauges[metric_name].set(float(latest["yhat_upper"]))

        if prediction.last_updated and last_updated_gauge:
            last_updated_gauge.labels(metric_name=metric_name).set(
                prediction.last_updated.timestamp()
            )


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None]:
    """Application lifespan manager."""
    global prediction_manager

    config = AppConfig()

    if config.metrics:
        setup_prometheus_metrics(config)

        prediction_manager = PredictionManager(
            prometheus_url=config.prometheus_url,
            metrics=config.metrics,
            update_interval_minutes=config.update_interval_minutes,
            hpo_enabled=config.hpo_enabled,
            hpo_n_trials=config.hpo_n_trials,
        )
        await prediction_manager.start()
        logger.info("Prediction manager started")
    else:
        logger.warning("No metrics configured, prediction manager not started")

    yield

    if prediction_manager:
        await prediction_manager.stop()
        logger.info("Prediction manager stopped")


app = FastAPI(
    title="Metrics Prediction Operator",
    description="Prophet-based metrics prediction service with Optuna HPO",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/")
async def root() -> dict[str, str]:
    """Root endpoint returning service info."""
    return {
        "service": "Metrics Prediction Operator",
        "version": "0.1.0",
        "status": "running",
    }


@app.get("/health")
async def health() -> dict[str, str]:
    """Health check endpoint."""
    return {"status": "healthy"}


@app.get("/metrics")
async def metrics() -> Response:
    """Prometheus metrics endpoint."""
    update_prometheus_metrics()
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST,
    )


@app.get("/predictions")
async def get_predictions() -> dict[str, Any]:
    """Get all current predictions."""
    global prediction_manager

    if prediction_manager is None:
        return {"error": "Prediction manager not initialized", "predictions": {}}

    predictions: dict[str, Any] = {}
    for metric_name, prediction in prediction_manager.predictions.items():
        predictions[metric_name] = {
            "is_training": prediction.is_training,
            "error_message": prediction.error_message,
            "last_updated": (
                prediction.last_updated.isoformat() if prediction.last_updated else None
            ),
            "data_points": len(prediction.data_df),
            "prediction_points": len(prediction.predictions_df),
            "current_prediction": prediction_manager.get_current_prediction_value(metric_name),
        }

    return {"predictions": predictions}


@app.get("/predictions/{metric_name}")
async def get_metric_prediction(metric_name: str) -> dict[str, Any]:
    """Get prediction for a specific metric."""
    global prediction_manager

    if prediction_manager is None:
        return {"error": "Prediction manager not initialized"}

    prediction = prediction_manager.get_prediction(metric_name)
    if prediction is None:
        return {"error": f"Metric '{metric_name}' not found"}

    # Convert predictions DataFrame to list of dicts for JSON response
    predictions_list = []
    if not prediction.predictions_df.empty:
        for _, row in prediction.predictions_df.iterrows():
            predictions_list.append({
                "timestamp": row["ds"].isoformat(),
                "yhat": float(row["yhat"]),
                "yhat_lower": float(row["yhat_lower"]),
                "yhat_upper": float(row["yhat_upper"]),
            })

    return {
        "metric_name": metric_name,
        "is_training": prediction.is_training,
        "error_message": prediction.error_message,
        "last_updated": (
            prediction.last_updated.isoformat() if prediction.last_updated else None
        ),
        "data_points": len(prediction.data_df),
        "current_prediction": prediction_manager.get_current_prediction_value(metric_name),
        "predictions": predictions_list,
    }


@app.post("/predictions/{metric_name}/refresh")
async def refresh_prediction(metric_name: str) -> dict[str, str]:
    """Trigger a refresh for a specific metric's predictions."""
    global prediction_manager

    if prediction_manager is None:
        return {"error": "Prediction manager not initialized"}

    prediction = prediction_manager.get_prediction(metric_name)
    if prediction is None:
        return {"error": f"Metric '{metric_name}' not found"}

    if prediction.is_training:
        return {"status": "already_training", "message": "Metric is already being updated"}

    # Trigger update in background
    import asyncio

    asyncio.create_task(prediction_manager.update_metric(metric_name))
    return {"status": "started", "message": f"Started updating predictions for {metric_name}"}
