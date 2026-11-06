"""MLflow and Arize observability setup for the agent."""

import os

import mlflow
from arize.otel import Transport, register
from openinference.instrumentation.langchain import LangChainInstrumentor

from tech_market_researcher.core.configuration import (
    ArizeObservabilityConfig,
    MlflowObservabilityConfig,
    ObservabilityConfig,
)
from tech_market_researcher.core.constants import PACKAGE_VERSION

MLFLOW_TRACKING_URI_ENV = "MLFLOW_SERVER_URI"
ARIZE_COLLECTOR_ENDPOINT_ENV = "ARIZE_COLLECTOR_ENDPOINT"

# arize-otel's register() requires space_id and api_key even for a local
# collector. Arize runs with auth disabled and ignores them, so these are
# local placeholders.
ARIZE_SPACE_ID = "local"
ARIZE_API_KEY = "local"


def setup_observability(config: ObservabilityConfig) -> None:
    """Configure MLflow tracking and Arize trace export, whichever is enabled."""
    if config.mlflow.enabled:
        _setup_mlflow(config.mlflow)
    if config.arize.enabled:
        _setup_arize(config.arize)


def _setup_mlflow(config: MlflowObservabilityConfig) -> None:
    """Configure MLflow tracking and auto-tracing."""
    tracking_uri = os.environ[MLFLOW_TRACKING_URI_ENV].rstrip("/")
    mlflow.set_tracking_uri(tracking_uri)
    version = PACKAGE_VERSION
    mlflow.set_experiment(config.experiment_name_template.format(project_version=version))
    mlflow.set_active_model(name=f"tech-market-researcher-v{version}".replace(".", "_"))
    # Enable auto-tracing for LangChain (and LangGraph-based frameworks like Deep Agent)
    mlflow.langchain.autolog()


def _setup_arize(config: ArizeObservabilityConfig) -> None:
    """Export LangChain spans to the local Arize server over OTLP gRPC.

    The collector endpoint (Arize's gRPC port, default :4317) comes from
    ARIZE_COLLECTOR_ENDPOINT, set in .env. OTLP HTTP is not used because
    arize-otel's HTTP exporter crashes in register()'s startup introspection.
    """
    tracer_provider = register(
        space_id=ARIZE_SPACE_ID,
        api_key=ARIZE_API_KEY,
        project_name=config.project_name,
        endpoint=os.environ[ARIZE_COLLECTOR_ENDPOINT_ENV],
        transport=Transport.GRPC,
        verbose=False,
    )
    LangChainInstrumentor().instrument(tracer_provider=tracer_provider)
