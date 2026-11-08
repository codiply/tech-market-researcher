"""MLflow tracking setup."""

import os

import mlflow

from tech_market_researcher.core.configuration import MlflowObservabilityConfig
from tech_market_researcher.core.constants import PROJECT_NAME

MLFLOW_TRACKING_URI_ENV = "MLFLOW_SERVER_URI"


def setup_mlflow(config: MlflowObservabilityConfig) -> None:
    """Configure MLflow tracking and auto-tracing."""
    tracking_uri = os.environ[MLFLOW_TRACKING_URI_ENV].rstrip("/")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(config.experiment_name)
    mlflow.set_active_model(name=PROJECT_NAME)
    # Enable auto-tracing for LangChain (and LangGraph-based frameworks like Deep Agent)
    mlflow.langchain.autolog()
