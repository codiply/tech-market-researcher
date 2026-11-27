"""Supervisor agent backed by the MLflow AI Gateway."""

import logging
from collections.abc import Callable, Sequence
from datetime import datetime, timezone
from typing import Any

from deepagents import create_deep_agent
from langchain_core.messages import BaseMessage
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph.state import CompiledStateGraph

from tech_market_researcher.capabilities.builders import build_middleware_list, build_tool_list
from tech_market_researcher.core.agent_backend import build_agent_backend
from tech_market_researcher.core.arguments import AgentCommandLineArguments
from tech_market_researcher.core.checkpointing import get_checkpointer
from tech_market_researcher.core.common import (
    AgentRuntimeContext,
    AsyncReleasableResource,
    SessionId,
    UserSession,
)
from tech_market_researcher.core.configuration import ApplicationConfig, load_config
from tech_market_researcher.core.constants import (
    AGENT_CHECKPOINT_THREAD_ID_KEY,
    AGENT_CHECKPOINT_USER_ID_METADATA_KEY,
)
from tech_market_researcher.core.debug_dumps import write_debug_dumps
from tech_market_researcher.core.logging_setup import setup_agent_logging
from tech_market_researcher.core.observability import agent_trace_context, setup_observability
from tech_market_researcher.core.skills import materialize_skill_bundles

logger = logging.getLogger(__name__)


class TechMarketResearcherAgent:
    """The agent, wrapping each invocation in MLflow tracing with session and agent metadata."""

    def __init__(
        self,
        agent: CompiledStateGraph[Any, Any],
        environment: str,
        async_releasable_resources: Sequence[AsyncReleasableResource] = (),
    ) -> None:
        self._agent = agent
        self._environment = environment
        self._async_releasable_resources = tuple(async_releasable_resources)

    async def ainvoke(
        self,
        messages: list[dict[str, str]],
        context: AgentRuntimeContext,
    ) -> dict[str, Any]:
        """Run one turn, tracing it under the session id and agent metadata.

        Prior conversation is resumed from the checkpointer under the context's thread id,
        so messages should only contain the new turns.
        """
        # The user id is stored in the checkpoint metadata so sessions can later be listed per user.
        config: RunnableConfig = {
            "configurable": {AGENT_CHECKPOINT_THREAD_ID_KEY: context.thread_id},
            "metadata": {AGENT_CHECKPOINT_USER_ID_METADATA_KEY: context.user_id},
        }
        # The AgentRuntimeContext constructor always generates a session id, so it is never None here.
        assert context.session_id is not None
        with agent_trace_context(context.session_id, self._environment):
            result = await self._agent.ainvoke({"messages": messages}, config=config, context=context, version="v1")
        logger.debug("Agent turn completed: session=%s, thread=%s", context.session_id, context.thread_id)
        return result

    async def __aenter__(self) -> "TechMarketResearcherAgent":
        """Enter the agent as an async context manager."""
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        """Exit the context, releasing the agent's async resources."""
        await self.close()

    async def close(self) -> None:
        """Release the agent's async resources, in the order they were provided."""
        for resource in self._async_releasable_resources:
            await resource.close()

    async def get_messages(self, context: AgentRuntimeContext) -> list[BaseMessage]:
        """Return the stored messages for the conversation under the context's thread id, if any."""
        config: RunnableConfig = {"configurable": {AGENT_CHECKPOINT_THREAD_ID_KEY: context.thread_id}}
        state = await self._agent.aget_state(config)
        messages = state.values.get("messages", [])
        logger.debug("Replaying %d stored message(s) for thread=%s", len(messages), context.thread_id)
        return messages

    async def alist_user_sessions(self, context: AgentRuntimeContext) -> list[UserSession]:
        """Return the sessions stored for the context's user.

        Sessions are matched on the user id tagged into the checkpoint metadata by ainvoke, so
        other users' sessions are never visible.
        """
        # create_agent always builds the agent with a checkpointer, and get_messages relies on it.
        assert isinstance(self._agent.checkpointer, BaseCheckpointSaver)
        prefix = f"{context.user_id}:"
        first_unseen = datetime.min.replace(tzinfo=timezone.utc)
        last_activity: dict[SessionId, datetime] = {}
        metadata_filter: dict[str, Any] = {AGENT_CHECKPOINT_USER_ID_METADATA_KEY: context.user_id}
        async for checkpoint_tuple in self._agent.checkpointer.alist(None, filter=metadata_filter):
            thread_id = checkpoint_tuple.config["configurable"][AGENT_CHECKPOINT_THREAD_ID_KEY]
            session_id: SessionId = thread_id[len(prefix) :]
            created_at = datetime.fromisoformat(checkpoint_tuple.checkpoint["ts"])
            if created_at > last_activity.get(session_id, first_unseen):
                last_activity[session_id] = created_at
        sessions = [
            UserSession(session_id=session_id, last_activity_at=ts) for session_id, ts in sorted(last_activity.items())
        ]
        logger.debug("Listed %d session(s) for user=%s", len(sessions), context.user_id)
        return sessions


def _apply_config_mutations(
    config: ApplicationConfig,
    hooks: list[Callable[[ApplicationConfig], None]] | None,
) -> ApplicationConfig:
    """Apply the config mutation hooks in place, then re-validate the mutated config."""
    for hook in hooks or ():
        hook(config)
    return ApplicationConfig.model_validate(config)


async def create_agent(
    args: AgentCommandLineArguments,
    config_mutations_hooks: list[Callable[[ApplicationConfig], None]] | None = None,
) -> TechMarketResearcherAgent:
    """Create the supervisor agent from the loaded config named by args.

    If given, the hooks in config_mutations_hooks mutate the loaded config in place before the
    agent is built. The returned agent is an async context manager: entering it binds the agent
    to the current event loop, and exiting it releases the agent's async resources (e.g. the
    connection pool).
    """
    config = load_config(args.config_name, args.extra_mixins)
    config = _apply_config_mutations(config, config_mutations_hooks)
    setup_agent_logging(config.logging)
    setup_observability(config.observability, config.environment)
    model_connection = config.get_model_connection_details(config.supervisor.model)
    logger.info(
        "Creating agent: config=%s, extra_mixins=%s, environment=%s, model=%s",
        args.config_name,
        args.extra_mixins,
        config.environment,
        model_connection.model_name,
    )
    model = ChatOpenAI(
        model=model_connection.model_name,
        base_url=model_connection.base_url,
        api_key=model_connection.api_key,
    )
    capabilities = config.supervisor.capabilities
    checkpointer_wrapper = await get_checkpointer(config.supervisor.checkpointer)
    checkpointer = checkpointer_wrapper.checkpointer
    backend, backend_resources = build_agent_backend(config.supervisor.backend)
    skills = materialize_skill_bundles(backend, config.supervisor.skill_bundles)
    agent = create_deep_agent(
        model=model,
        system_prompt=config.supervisor.system_prompt,
        tools=await build_tool_list(config),
        middleware=build_middleware_list(capabilities),
        context_schema=AgentRuntimeContext,
        checkpointer=checkpointer,
        skills=skills or None,
        backend=backend,
    )
    write_debug_dumps(config.debug, agent)
    return TechMarketResearcherAgent(
        agent,
        config.environment,
        async_releasable_resources=(checkpointer_wrapper, *backend_resources),
    )
