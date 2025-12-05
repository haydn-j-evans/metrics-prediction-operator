"""Main entry point for the metrics prediction operator."""

import logging
import sys

import uvicorn

from .config import AppConfig


def setup_logging() -> None:
    """Set up logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    # Reduce noise from some libraries
    logging.getLogger("cmdstanpy").setLevel(logging.WARNING)
    logging.getLogger("prophet").setLevel(logging.WARNING)
    logging.getLogger("optuna").setLevel(logging.WARNING)


def main() -> None:
    """Main entry point."""
    setup_logging()

    config = AppConfig()

    uvicorn.run(
        "metrics_prediction_operator.app:app",
        host=config.host,
        port=config.port,
        log_level="info",
    )


if __name__ == "__main__":
    main()
