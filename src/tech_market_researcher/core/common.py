"""Shared models and helpers used across the package."""

from pydantic import BaseModel


class AgentContext(BaseModel):
    """Per-run context passed to agents on every invocation."""

    user_id: str
    session_id: str
    time_zone: str
