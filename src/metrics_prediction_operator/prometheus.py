"""Prometheus data fetching and processing."""

import logging
from datetime import UTC, datetime, timedelta

import httpx
import pandas as pd

logger = logging.getLogger(__name__)


async def fetch_prometheus_data(
    prometheus_url: str,
    query: str,
    start_time: datetime | None = None,
    end_time: datetime | None = None,
    step: str = "1h",
) -> pd.DataFrame:
    """Fetch time series data from Prometheus.

    Args:
        prometheus_url: URL of the Prometheus server.
        query: PromQL query to execute.
        start_time: Start time for the range query. Defaults to 7 days ago.
        end_time: End time for the range query. Defaults to now.
        step: Query resolution step (e.g., '1h', '5m').

    Returns:
        DataFrame with 'ds' and 'y' columns suitable for Prophet.
    """
    now = datetime.now(tz=UTC)

    if end_time is None:
        end_time = now
    if start_time is None:
        start_time = end_time - timedelta(days=7)

    url = f"{prometheus_url}/api/v1/query_range"
    params: dict[str, str | float] = {
        "query": query,
        "start": start_time.timestamp(),
        "end": end_time.timestamp(),
        "step": step,
    }

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(url, params=params)
            response.raise_for_status()
            data = response.json()

        if data["status"] != "success":
            logger.error(f"Prometheus query failed: {data.get('error', 'Unknown error')}")
            return pd.DataFrame(columns=["ds", "y"])

        result = data.get("data", {}).get("result", [])
        if not result:
            logger.warning(f"No data returned for query: {query}")
            return pd.DataFrame(columns=["ds", "y"])

        # Extract values from the first result
        values = result[0].get("values", [])
        if not values:
            return pd.DataFrame(columns=["ds", "y"])

        # Convert to DataFrame
        df = pd.DataFrame(values, columns=["timestamp", "value"])
        df["ds"] = pd.to_datetime(df["timestamp"], unit="s", utc=True)
        df["y"] = pd.to_numeric(df["value"], errors="coerce")
        df = df[["ds", "y"]].dropna()

        logger.info(f"Fetched {len(df)} data points for query: {query}")
        return df

    except httpx.HTTPError as e:
        logger.error(f"HTTP error fetching Prometheus data: {e}")
        return pd.DataFrame(columns=["ds", "y"])
    except Exception as e:
        logger.error(f"Error fetching Prometheus data: {e}")
        return pd.DataFrame(columns=["ds", "y"])


def process_dataframe_for_prophet(df: pd.DataFrame) -> pd.DataFrame:
    """Process a DataFrame to ensure it's suitable for Prophet.

    Args:
        df: DataFrame with 'ds' and 'y' columns.

    Returns:
        Cleaned DataFrame ready for Prophet training.
    """
    if df.empty:
        return df

    # Ensure proper types
    df = df.copy()
    df["ds"] = pd.to_datetime(df["ds"])
    # Prophet requires timezone-naive timestamps
    if df["ds"].dt.tz is not None:
        df["ds"] = df["ds"].dt.tz_localize(None)
    df["y"] = pd.to_numeric(df["y"], errors="coerce")

    # Remove any NaN values
    df = df.dropna()

    # Sort by timestamp
    df = df.sort_values("ds").reset_index(drop=True)

    # Remove duplicates by timestamp (keep last)
    df = df.drop_duplicates(subset=["ds"], keep="last")

    return df
