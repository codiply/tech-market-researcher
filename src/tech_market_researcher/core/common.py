"""Shared models and helpers used across the package."""

from pydantic import BaseModel, ConfigDict


class BasePydanticModel(BaseModel):
    """Base for all pydantic models; unknown fields are rejected."""

    model_config = ConfigDict(extra="forbid")


class AgentRuntimeContext(BaseModel):
    """Per-run runtime context passed to agents on every invocation."""

    user_id: str
    conversation_id: str
    time_zone: str
