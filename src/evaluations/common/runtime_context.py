"""Runtime context construction shared by the evaluation platforms."""

from tech_market_researcher.core.common import AgentRuntimeContext
from tech_market_researcher.core.constants import DEFAULT_CONVERSATION_ID, DEFAULT_TIME_ZONE, DEFAULT_USER_ID

# Defaults for every required AgentRuntimeContext field; the record's context
# overrides them. New fields with their own pydantic defaults are filled in by the model.
DEFAULT_RUNTIME_CONTEXT = {
    "user_id": DEFAULT_USER_ID,
    "conversation_id": DEFAULT_CONVERSATION_ID,
    "time_zone": DEFAULT_TIME_ZONE,
}


def build_runtime_context(dataset_record_context: dict[str, str]) -> AgentRuntimeContext:
    """Build the runtime context for a dataset record, applying defaults for missing fields."""
    # Unknown keys are ignored by the model, so extra record fields are safe to pass through.
    return AgentRuntimeContext.model_validate({**DEFAULT_RUNTIME_CONTEXT, **dataset_record_context})
