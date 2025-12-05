"""Tests for configuration module."""


from metrics_prediction_operator.config import AppConfig, MetricConfig


def test_metric_config_defaults() -> None:
    """Test MetricConfig default values."""
    config = MetricConfig(name="test_metric", query='rate(http_requests_total[5m])')
    assert config.name == "test_metric"
    assert config.query == 'rate(http_requests_total[5m])'
    assert config.prediction_horizon_hours == 24


def test_metric_config_custom_horizon() -> None:
    """Test MetricConfig with custom prediction horizon."""
    config = MetricConfig(
        name="test_metric",
        query='rate(http_requests_total[5m])',
        prediction_horizon_hours=48,
    )
    assert config.prediction_horizon_hours == 48


def test_app_config_defaults() -> None:
    """Test AppConfig default values."""
    config = AppConfig()
    assert config.prometheus_url == "http://localhost:9090"
    assert config.update_interval_minutes == 30
    assert config.host == "0.0.0.0"
    assert config.port == 8000
    assert config.hpo_enabled is True
    assert config.hpo_n_trials == 50
    assert config.metrics == []


def test_app_config_with_metrics() -> None:
    """Test AppConfig with metrics list."""
    metrics = [
        MetricConfig(name="metric1", query="query1"),
        MetricConfig(name="metric2", query="query2"),
    ]
    config = AppConfig(metrics=metrics)
    assert len(config.metrics) == 2
    assert config.metrics[0].name == "metric1"
    assert config.metrics[1].name == "metric2"
