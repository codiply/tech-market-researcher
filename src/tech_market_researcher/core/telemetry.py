"""MLflow telemetry setup."""

import os

import mlflow

TRACKING_URI_ENV = "MLFLOW_TRACKING_URI"
EXPERIMENT_ENV = "MLFLOW_EXPERIMENT"


def setup_telemetry(version: str) -> None:
    """Configure MLflow tracking and auto-tracing for the given version."""
    tracking_uri = os.environ[TRACKING_URI_ENV].rstrip("/")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(os.environ[EXPERIMENT_ENV])
    mlflow.set_active_model(name=f"tech-market-researcher-v{version}".replace(".", "_"))
    # Enable auto-tracing for LangChain (and LangGraph-based frameworks like Deep Agent)
    mlflow.langchain.autolog()
