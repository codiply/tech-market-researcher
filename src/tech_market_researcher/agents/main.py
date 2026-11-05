"""Supervisor agent backed by the MLflow AI Gateway."""

from deepagents import create_deep_agent
from langchain_openai import ChatOpenAI

from tech_market_researcher.capabilities.builders import build_middleware_list, build_tool_list
from tech_market_researcher.core.arguments import AgentCommandLineArguments
from tech_market_researcher.core.common import AgentRuntimeContext
from tech_market_researcher.core.configuration import load_config
from tech_market_researcher.core.observability import setup_observability


def create_agent(args: AgentCommandLineArguments):
    """Create the supervisor agent from the loaded config named by args."""
    config = load_config(args.config_name)
    setup_observability(config.observability)
    model_connection = config.get_model_connection_details(config.supervisor.model)
    model = ChatOpenAI(
        model=model_connection.model_name,
        base_url=model_connection.base_url,
        api_key=model_connection.api_key,
    )
    capabilities = config.supervisor.capabilities
    return create_deep_agent(
        model=model,
        system_prompt=config.supervisor.system_prompt,
        tools=build_tool_list(capabilities),
        middleware=build_middleware_list(capabilities),
        context_schema=AgentRuntimeContext,
    )
