"""Tests for Prophet model module."""


from metrics_prediction_operator.model import (
    DEFAULT_PARAMS,
    create_prophet_model,
    create_sample_dataframe,
    generate_predictions,
    train_model,
)


def test_create_prophet_model_default() -> None:
    """Test creating a Prophet model with default parameters."""
    model = create_prophet_model()
    assert model is not None
    assert model.changepoint_prior_scale == DEFAULT_PARAMS["changepoint_prior_scale"]


def test_create_prophet_model_custom_params() -> None:
    """Test creating a Prophet model with custom parameters."""
    custom_params = {"changepoint_prior_scale": 0.1}
    model = create_prophet_model(custom_params)
    assert model.changepoint_prior_scale == 0.1


def test_create_sample_dataframe() -> None:
    """Test sample DataFrame creation."""
    df = create_sample_dataframe()
    assert not df.empty
    assert "ds" in df.columns
    assert "y" in df.columns
    assert len(df) == 168  # 7 days of hourly data


def test_train_model_without_hpo() -> None:
    """Test training a model without HPO."""
    df = create_sample_dataframe()
    model = train_model(df, use_hpo=False)
    assert model is not None


def test_train_model_with_small_dataset() -> None:
    """Test that HPO falls back to defaults with small dataset."""
    df = create_sample_dataframe().head(10)
    model = train_model(df, use_hpo=True, n_trials=2)
    assert model is not None


def test_generate_predictions() -> None:
    """Test prediction generation."""
    df = create_sample_dataframe()
    model = train_model(df, use_hpo=False)
    predictions = generate_predictions(model, periods=24, freq="h")

    assert not predictions.empty
    assert "ds" in predictions.columns
    assert "yhat" in predictions.columns
    assert "yhat_lower" in predictions.columns
    assert "yhat_upper" in predictions.columns
