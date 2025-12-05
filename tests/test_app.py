"""Tests for FastAPI application."""

import pytest
from fastapi.testclient import TestClient

from metrics_prediction_operator.app import app


@pytest.fixture
def client() -> TestClient:
    """Create a test client for the FastAPI app."""
    return TestClient(app)


def test_root_endpoint(client: TestClient) -> None:
    """Test the root endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["service"] == "Metrics Prediction Operator"
    assert data["version"] == "0.1.0"
    assert data["status"] == "running"


def test_health_endpoint(client: TestClient) -> None:
    """Test the health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"


def test_metrics_endpoint(client: TestClient) -> None:
    """Test the Prometheus metrics endpoint."""
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "text/plain" in response.headers["content-type"]


def test_predictions_endpoint_no_manager(client: TestClient) -> None:
    """Test predictions endpoint when manager is not initialized."""
    response = client.get("/predictions")
    assert response.status_code == 200
    data = response.json()
    # Without metrics configured, manager won't be initialized
    assert "predictions" in data or "error" in data


def test_prediction_not_found(client: TestClient) -> None:
    """Test getting prediction for non-existent metric."""
    response = client.get("/predictions/nonexistent_metric")
    assert response.status_code == 200
    data = response.json()
    assert "error" in data
