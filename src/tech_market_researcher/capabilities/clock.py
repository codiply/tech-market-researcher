"""Tool that reports the current datetime in the user's configured timezone."""

from datetime import datetime, timezone, tzinfo
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from langchain.tools import ToolRuntime, tool

from tech_market_researcher.core.common import AgentRuntimeContext


@tool
def current_datetime(runtime: ToolRuntime[AgentRuntimeContext]) -> str:
    """Return the current date and time in the user's timezone, with its UTC offset.

    Falls back to UTC if the configured timezone cannot be parsed."""
    zone: tzinfo
    try:
        zone = ZoneInfo(runtime.context.time_zone)
    except ZoneInfoNotFoundError:
        zone = timezone.utc
    now = datetime.now(zone)
    return f"{now.isoformat()} ({runtime.context.time_zone})"
