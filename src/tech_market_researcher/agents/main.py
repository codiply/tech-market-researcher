"""Supervisor agent backed by the MLflow AI Gateway."""

from collections.abc import Callable
from typing import Any

from deepagents import create_deep_agent
from langchain_openai import ChatOpenAI
from langgraph.graph.state import CompiledStateGraph

from tech_market_researcher.capabilities.builders import build_middleware_list, build_tool_list
from tech_market_researcher.core.arguments import AgentCommandLineArguments
from tech_market_researcher.core.common import AgentRuntimeContext
from tech_market_researcher.core.configuration import ApplicationConfig, load_config
from tech_market_researcher.core.observability import agent_trace_context, setup_observability


# pylint: disable=too-few-public-methods
class TechMarketResearcherAgent:
    """The agent, wrapping each invocation in MLflow tracing with session and agent metadata."""

    def __init__(self, agent: CompiledStateGraph[Any, Any], environment: str) -> None:
        self._agent = agent
        self._environment = environment

    def invoke(self, messages: list[dict[str, str]], context: AgentRuntimeContext) -> dict[str, Any]:
        """Run one turn, tracing it under the session id and agent metadata."""
        with agent_trace_context(context.session_id, self._environment):
            return self._agent.invoke({"messages": messages}, context=context)


def create_agent(
    args: AgentCommandLineArguments,
    config_mutations_hook: Callable[[ApplicationConfig], None] | None = None,
) -> TechMarketResearcherAgent:
    """Create the supervisor agent from the loaded config named by args.

    If given, config_mutations_hook mutates the loaded config in place before the agent is built.
    """
    config = load_config(args.config_name)
    if config_mutations_hook is not None:
        config_mutations_hook(config)
    setup_observability(config.observability)
    model_connection = config.get_model_connection_details(config.supervisor.model)
    model = ChatOpenAI(
        model=model_connection.model_name,
        base_url=model_connection.base_url,
        api_key=model_connection.api_key,
    )
    capabilities = config.supervisor.capabilities
    agent = create_deep_agent(
        model=model,
        system_prompt=config.supervisor.system_prompt,
        tools=build_tool_list(capabilities),
        middleware=build_middleware_list(capabilities),
        context_schema=AgentRuntimeContext,
    )
    return TechMarketResearcherAgent(agent, config.environment)
