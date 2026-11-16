"""Supervisor agent backed by the MLflow AI Gateway."""

from collections.abc import Callable, Sequence
from typing import Any

from deepagents import create_deep_agent
from langchain_core.messages import BaseMessage
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
from langgraph.graph.state import CompiledStateGraph

from tech_market_researcher.capabilities.builders import build_middleware_list, build_tool_list
from tech_market_researcher.core.arguments import AgentCommandLineArguments
from tech_market_researcher.core.checkpointing import get_checkpointer
from tech_market_researcher.core.common import AgentRuntimeContext, AsyncReleasableResource
from tech_market_researcher.core.configuration import ApplicationConfig, load_config
from tech_market_researcher.core.observability import agent_trace_context, setup_observability


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
        config: RunnableConfig = {"configurable": {"thread_id": context.thread_id}}
        # The AgentRuntimeContext constructor always generates a session id, so it is never None here.
        assert context.session_id is not None
        with agent_trace_context(context.session_id, self._environment):
            return await self._agent.ainvoke({"messages": messages}, config=config, context=context, version="v1")

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
        config: RunnableConfig = {"configurable": {"thread_id": context.thread_id}}
        state = await self._agent.aget_state(config)
        return state.values.get("messages", [])


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
    config = load_config(args.config_name)
    config = _apply_config_mutations(config, config_mutations_hooks)
    setup_observability(config.observability, config.environment)
    model_connection = config.get_model_connection_details(config.supervisor.model)
    model = ChatOpenAI(
        model=model_connection.model_name,
        base_url=model_connection.base_url,
        api_key=model_connection.api_key,
    )
    capabilities = config.supervisor.capabilities
    checkpointer_wrapper = await get_checkpointer(config.supervisor.checkpointer)
    checkpointer = checkpointer_wrapper.checkpointer
    agent = create_deep_agent(
        model=model,
        system_prompt=config.supervisor.system_prompt,
        tools=await build_tool_list(config),
        middleware=build_middleware_list(capabilities),
        context_schema=AgentRuntimeContext,
        checkpointer=checkpointer,
    )
    return TechMarketResearcherAgent(
        agent,
        config.environment,
        async_releasable_resources=(checkpointer_wrapper,),
    )
