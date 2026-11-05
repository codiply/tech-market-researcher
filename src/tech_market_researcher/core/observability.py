"""MLflow and Arize observability setup for the agent."""

import os

import mlflow
from arize.otel import Transport, register
from openinference.instrumentation.langchain import LangChainInstrumentor

MLFLOW_TRACKING_URI_ENV = "MLFLOW_SERVER_URI"
MLFLOW_EXPERIMENT_ENV = "MLFLOW_EXPERIMENT"
ARIZE_COLLECTOR_ENDPOINT_ENV = "ARIZE_COLLECTOR_ENDPOINT"
ARIZE_PROJECT_ENV = "ARIZE_PROJECT_NAME"

# arize-otel's register() requires space_id and api_key even for a local
# collector. Arize runs with auth disabled and ignores them, so these are
# local placeholders.
ARIZE_SPACE_ID = "local"
ARIZE_API_KEY = "local"
DEFAULT_ARIZE_PROJECT = "tech-market-researcher"


def setup_observability(version: str) -> None:
    """Configure MLflow tracking and Arize trace export for the agent.

    MLflow keeps its auto-tracing for the gateway and model registry; Arize
    receives the same LangChain spans over OTLP for the observability UI.
    """
    _setup_mlflow(version)
    _setup_arize()


def _setup_mlflow(version: str) -> None:
    """Configure MLflow tracking and auto-tracing for the given version."""
    tracking_uri = os.environ[MLFLOW_TRACKING_URI_ENV].rstrip("/")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(os.environ[MLFLOW_EXPERIMENT_ENV])
    mlflow.set_active_model(name=f"tech-market-researcher-v{version}".replace(".", "_"))
    # Enable auto-tracing for LangChain (and LangGraph-based frameworks like Deep Agent)
    mlflow.langchain.autolog()


def _setup_arize() -> None:
    """Export LangChain spans to the local Arize server over OTLP gRPC.

    The collector endpoint (Arize's gRPC port, default :4317) comes from
    ARIZE_COLLECTOR_ENDPOINT and the project name from ARIZE_PROJECT_NAME, both
    set in .env. OTLP HTTP is not used because arize-otel's HTTP exporter
    crashes in register()'s startup introspection.
    """
    tracer_provider = register(
        space_id=ARIZE_SPACE_ID,
        api_key=ARIZE_API_KEY,
        project_name=os.environ.get(ARIZE_PROJECT_ENV, DEFAULT_ARIZE_PROJECT),
        endpoint=os.environ[ARIZE_COLLECTOR_ENDPOINT_ENV],
        transport=Transport.GRPC,
        verbose=False,
    )
    LangChainInstrumentor().instrument(tracer_provider=tracer_provider)
