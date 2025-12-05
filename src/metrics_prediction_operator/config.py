"""Configuration module for the metrics prediction operator."""

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class MetricConfig(BaseSettings):
    """Configuration for a single Prometheus metric to predict."""

    model_config = SettingsConfigDict(extra="forbid")

    name: str = Field(description="Name of the Prometheus metric")
    query: str = Field(description="PromQL query to fetch the metric data")
    prediction_horizon_hours: int = Field(
        default=24, description="Number of hours to predict into the future"
    )


class AppConfig(BaseSettings):
    """Application configuration loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_prefix="MPO_",
        env_nested_delimiter="__",
        extra="forbid",
    )

    # Prometheus configuration
    prometheus_url: str = Field(
        default="http://localhost:9090",
        description="URL of the Prometheus server",
    )

    # Metrics to predict
    metrics: list[MetricConfig] = Field(
        default_factory=list,
        description="List of metrics to predict",
    )

    # Update interval
    update_interval_minutes: int = Field(
        default=30,
        description="Interval in minutes to update predictions",
    )

    # Server configuration
    host: str = Field(default="0.0.0.0", description="Host to bind the server to")
    port: int = Field(default=8000, description="Port to bind the server to")

    # Optuna HPO configuration
    hpo_n_trials: int = Field(
        default=50,
        description="Number of Optuna trials for hyperparameter optimization",
    )
    hpo_enabled: bool = Field(
        default=True,
        description="Enable hyperparameter optimization with Optuna",
    )
