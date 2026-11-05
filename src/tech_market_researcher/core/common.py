"""Shared models and helpers used across the package."""

from pydantic import BaseModel


class AgentRuntimeContext(BaseModel):
    """Per-run runtime context passed to agents on every invocation."""

    user_id: str
    conversation_id: str
    time_zone: str
