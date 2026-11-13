"""MLflow and Phoenix observability setup for the agent."""

from contextlib import contextmanager
from typing import Generator

import mlflow as _mlflow
from openinference.instrumentation import using_session

from tech_market_researcher.core.configuration import ObservabilityConfig
from tech_market_researcher.core.constants import PACKAGE_VERSION
from tech_market_researcher.core.observability.mlflow import setup_mlflow
from tech_market_researcher.core.observability.phoenix import setup_phoenix

__all__ = ["agent_trace_context", "setup_observability"]


@contextmanager
def agent_trace_context(session_id: str, environment: str) -> Generator[None, None, None]:
    """Tag every trace created in the scope with the session id and agent metadata.

    The session id is propagated through the OpenTelemetry context, so the
    OpenInference tracer stamps it as ``session.id`` on the Phoenix spans.
    """
    with (
        using_session(session_id),
        _mlflow.tracing.context(
            session_id=session_id,
            tags={"agent_version": PACKAGE_VERSION, "agent_environment": environment},
        ),
    ):
        yield


def setup_observability(config: ObservabilityConfig, environment: str) -> None:
    """Configure MLflow tracking and Phoenix trace export, whichever is enabled."""
    if config.mlflow.enabled:
        setup_mlflow(config.mlflow)
    if config.phoenix.enabled:
        setup_phoenix(config.phoenix, environment)
