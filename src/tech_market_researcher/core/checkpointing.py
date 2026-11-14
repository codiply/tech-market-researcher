"""Checkpointer factories."""

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import MemorySaver

from tech_market_researcher.core.configuration import CheckpointerBackend, CheckpointerConfig


def get_checkpointer(config: CheckpointerConfig) -> BaseCheckpointSaver:
    """Create the checkpointer selected by config."""
    match config.backend:
        case CheckpointerBackend.MEMORY:
            return MemorySaver()
