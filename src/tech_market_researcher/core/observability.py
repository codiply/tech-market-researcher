"""MLflow and Arize observability setup for the agent."""

import os
from contextlib import contextmanager
from typing import Generator

import mlflow
from arize.otel import Transport, register
from openinference.instrumentation.langchain import LangChainInstrumentor
from opentelemetry import context as otel_context
from opentelemetry.sdk.trace import ReadableSpan, Span, SpanProcessor

from tech_market_researcher.core.configuration import (
    ArizeObservabilityConfig,
    MlflowObservabilityConfig,
    ObservabilityConfig,
)
from tech_market_researcher.core.constants import PACKAGE_VERSION, PROJECT_NAME

MLFLOW_TRACKING_URI_ENV = "MLFLOW_SERVER_URI"
ARIZE_COLLECTOR_ENDPOINT_ENV = "ARIZE_COLLECTOR_ENDPOINT"

# arize-otel's register() requires space_id and api_key even for a local
# collector. Arize runs with auth disabled and ignores them, so these are
# local placeholders.
ARIZE_SPACE_ID = "local"
ARIZE_API_KEY = "local"


class AgentAttributeSpanProcessor(SpanProcessor):
    """Stamp process-level agent attributes onto every span before export."""

    def __init__(self, attributes: dict[str, str]) -> None:
        self._attributes = attributes

    def on_start(
        self,
        span: Span,
        parent_context: otel_context.Context | None = None,
    ) -> None:
        for key, value in self._attributes.items():
            span.set_attribute(key, value)

    def on_end(self, span: ReadableSpan) -> None:
        pass

    def shutdown(self) -> None:
        pass

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        return True


@contextmanager
def agent_trace_context(session_id: str, environment: str) -> Generator[None, None, None]:
    """Tag every trace created in the scope with the session id and agent metadata."""
    with mlflow.tracing.context(
        session_id=session_id,
        tags={"agent_version": PACKAGE_VERSION, "agent_environment": environment},
    ):
        yield


def setup_observability(config: ObservabilityConfig, environment: str) -> None:
    """Configure MLflow tracking and Arize trace export, whichever is enabled."""
    if config.mlflow.enabled:
        _setup_mlflow(config.mlflow)
    if config.arize.enabled:
        _setup_arize(config.arize, environment)


def _setup_mlflow(config: MlflowObservabilityConfig) -> None:
    """Configure MLflow tracking and auto-tracing."""
    tracking_uri = os.environ[MLFLOW_TRACKING_URI_ENV].rstrip("/")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(config.experiment_name)
    mlflow.set_active_model(name=PROJECT_NAME)
    # Enable auto-tracing for LangChain (and LangGraph-based frameworks like Deep Agent)
    mlflow.langchain.autolog()


def _setup_arize(config: ArizeObservabilityConfig, environment: str) -> None:
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
        span_processors=[
            AgentAttributeSpanProcessor(
                {
                    "agent_version": PACKAGE_VERSION,
                    "agent_environment": environment,
                }
            )
        ],
    )
    LangChainInstrumentor().instrument(tracer_provider=tracer_provider)
