# Metrics Prediction Operator

Uses Prophet to generate prediction metrics in order to detect anomalies.

## Features

- **Prophet-based Predictions**: Uses Facebook's Prophet library for time series forecasting
- **Optuna HPO**: Automatic hyperparameter optimization using Optuna
- **Prometheus Integration**: Fetches metrics from Prometheus and exposes predictions as Prometheus metrics
- **FastAPI Application**: RESTful API for managing and querying predictions
- **Configurable**: Pydantic-based configuration with environment variable support

## Requirements

- Python 3.14+
- uv package manager

## Installation

```bash
# Clone the repository
git clone https://github.com/haydn-j-evans/metrics-prediction-operator.git
cd metrics-prediction-operator

# Install dependencies
uv sync
```

## Configuration

The application is configured via environment variables with the `MPO_` prefix:

| Variable | Default | Description |
|----------|---------|-------------|
| `MPO_PROMETHEUS_URL` | `http://localhost:9090` | URL of the Prometheus server |
| `MPO_UPDATE_INTERVAL_MINUTES` | `30` | Interval between prediction updates |
| `MPO_HOST` | `0.0.0.0` | Host to bind the server to |
| `MPO_PORT` | `8000` | Port to bind the server to |
| `MPO_HPO_ENABLED` | `true` | Enable Optuna hyperparameter optimization |
| `MPO_HPO_N_TRIALS` | `50` | Number of Optuna trials for HPO |

### Configuring Metrics

Metrics are configured as a JSON list in the `MPO_METRICS` environment variable:

```bash
export MPO_METRICS='[
  {
    "name": "http_requests_rate",
    "query": "rate(http_requests_total[5m])",
    "prediction_horizon_hours": 24
  },
  {
    "name": "cpu_usage",
    "query": "avg(rate(node_cpu_seconds_total{mode!=\"idle\"}[5m]))",
    "prediction_horizon_hours": 48
  }
]'
```

## Usage

### Running the Server

```bash
# Using uv
uv run metrics-prediction-operator

# Or directly with Python
uv run python -m metrics_prediction_operator.main
```

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Service info |
| `/health` | GET | Health check |
| `/metrics` | GET | Prometheus metrics endpoint |
| `/predictions` | GET | Get all current predictions |
| `/predictions/{metric_name}` | GET | Get prediction for a specific metric |
| `/predictions/{metric_name}/refresh` | POST | Trigger a refresh for a metric's predictions |

### Prometheus Metrics

The application exposes the following Prometheus metrics for each configured metric:

- `{metric_name}_prediction`: The predicted value
- `{metric_name}_prediction_lower`: Lower bound of the prediction interval
- `{metric_name}_prediction_upper`: Upper bound of the prediction interval
- `prediction_last_updated_timestamp`: Timestamp of the last prediction update

## Development

### Setting up the Development Environment

```bash
# Install all dependencies including dev dependencies
uv sync --all-groups

# Run linting
uv run ruff check .

# Run type checking
uv run mypy src/metrics_prediction_operator

# Run tests
uv run pytest tests/ -v
```

### Project Structure

```
metrics-prediction-operator/
├── src/
│   └── metrics_prediction_operator/
│       ├── __init__.py        # Package exports
│       ├── app.py             # FastAPI application
│       ├── config.py          # Pydantic configuration
│       ├── main.py            # Entry point
│       ├── model.py           # Prophet model with Optuna HPO
│       ├── prediction_manager.py  # Prediction lifecycle management
│       └── prometheus.py      # Prometheus data fetching
├── tests/
│   ├── test_app.py
│   ├── test_config.py
│   ├── test_model.py
│   └── test_prometheus.py
├── pyproject.toml
└── README.md
```

## License

MIT License - see [LICENSE](LICENSE) for details.
