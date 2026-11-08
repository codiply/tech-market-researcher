"""Arize trace export over OTLP gRPC."""

import os

from arize.otel import Transport, register
from openinference.instrumentation.langchain import LangChainInstrumentor
from opentelemetry import context as otel_context
from opentelemetry.sdk.trace import ReadableSpan, Span, SpanProcessor

from tech_market_researcher.core.configuration import ArizeObservabilityConfig
from tech_market_researcher.core.constants import PACKAGE_VERSION

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


def setup_arize(config: ArizeObservabilityConfig, environment: str) -> None:
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
