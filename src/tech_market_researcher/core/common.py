"""Shared models and helpers used across the package."""

from typing import Any
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator

# User ids must start with a letter, then contain only alphanumeric characters, underscores and hyphens.
USER_ID_PATTERN = r"^[A-Za-z][A-Za-z0-9_-]*$"
# Session ids may only contain alphanumeric characters, underscores and hyphens.
SESSION_ID_PATTERN = r"^[A-Za-z0-9_-]+$"


class BasePydanticModel(BaseModel):
    """Base for all pydantic models; unknown fields are rejected."""

    model_config = ConfigDict(extra="forbid")


class AgentRuntimeContext(BaseModel):
    """Per-run runtime context passed to agents on every invocation."""

    user_id: str = Field(pattern=USER_ID_PATTERN)
    session_id: str | None = None
    time_zone: str

    @model_validator(mode="before")
    @classmethod
    def _generate_new_session_id(cls, data: Any) -> Any:
        """Generate a fresh session id when it is omitted or passed as None, so each run starts its own thread."""
        if isinstance(data, dict) and data.get("session_id") is None:
            data = {**data, "session_id": str(uuid4())}
        return data

    @property
    def thread_id(self) -> str:
        """Conversation key."""
        return f"{self.user_id}:{self.session_id}"
