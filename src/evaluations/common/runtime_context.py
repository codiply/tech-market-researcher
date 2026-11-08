"""Runtime context construction shared by the evaluation platforms."""

from uuid import uuid4

from evaluations.common.constants import EVALS_SESSION_ID_PREFIX, EVALS_USER_ID
from tech_market_researcher.core.common import AgentRuntimeContext
from tech_market_researcher.core.constants import DEFAULT_TIME_ZONE

# Defaults for every required AgentRuntimeContext field; the record's context
# overrides them. New fields with their own pydantic defaults are filled in by the model.
DEFAULT_RUNTIME_CONTEXT = {
    "user_id": EVALS_USER_ID,
    "time_zone": DEFAULT_TIME_ZONE,
}


def build_runtime_context(dataset_record_context: dict[str, str]) -> AgentRuntimeContext:
    """Build the runtime context for a dataset record, applying defaults for missing fields."""
    context = {**DEFAULT_RUNTIME_CONTEXT, **dataset_record_context}
    # Each record runs as its own session unless the record provides a session id.
    context.setdefault("session_id", f"{EVALS_SESSION_ID_PREFIX}{uuid4()}")
    # Unknown keys are ignored by the model, so extra record fields are safe to pass through.
    return AgentRuntimeContext.model_validate(context)
