"""Tests for Prometheus data fetching."""

import pandas as pd

from metrics_prediction_operator.prometheus import process_dataframe_for_prophet


def test_process_empty_dataframe() -> None:
    """Test processing an empty DataFrame."""
    df = pd.DataFrame(columns=["ds", "y"])
    result = process_dataframe_for_prophet(df)
    assert result.empty


def test_process_dataframe_with_data() -> None:
    """Test processing a DataFrame with valid data."""
    df = pd.DataFrame({
        "ds": pd.date_range("2024-01-01", periods=10, freq="h"),
        "y": range(10),
    })
    result = process_dataframe_for_prophet(df)

    assert len(result) == 10
    assert result["ds"].is_monotonic_increasing


def test_process_dataframe_removes_duplicates() -> None:
    """Test that duplicate timestamps are removed."""
    df = pd.DataFrame({
        "ds": ["2024-01-01 00:00", "2024-01-01 00:00", "2024-01-01 01:00"],
        "y": [1, 2, 3],
    })
    result = process_dataframe_for_prophet(df)
    assert len(result) == 2


def test_process_dataframe_handles_nan() -> None:
    """Test that NaN values are removed."""
    df = pd.DataFrame({
        "ds": pd.date_range("2024-01-01", periods=5, freq="h"),
        "y": [1, 2, None, 4, 5],
    })
    result = process_dataframe_for_prophet(df)
    assert len(result) == 4
