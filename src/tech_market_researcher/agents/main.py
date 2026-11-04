"""Supervisor agent backed by the MLflow AI Gateway."""

from deepagents import create_deep_agent
from langchain_openai import ChatOpenAI

from tech_market_researcher.capabilities.builders import build_middleware_list, build_tool_list
from tech_market_researcher.core.configuration import ApplicationConfig


def create_agent(config: ApplicationConfig):
    """Create the supervisor agent."""
    model_connection = config.get_model_connection_details(config.supervisor.model)
    model = ChatOpenAI(
        model=model_connection.model_name,
        base_url=model_connection.base_url,
        api_key=model_connection.api_key,
    )
    capabilities = config.supervisor.capabilities
    return create_deep_agent(
        model=model,
        system_prompt="You are a Researcher giving me insights into the Tech Market.",
        tools=build_tool_list(capabilities),
        middleware=build_middleware_list(capabilities),
    )
