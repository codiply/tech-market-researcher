"""Supervisor agent backed by the MLflow AI Gateway."""

from deepagents import create_deep_agent
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

from tech_market_researcher.core.configuration import ApplicationConfig


def create_agent(config: ApplicationConfig):
    """Create the supervisor agent."""
    model_config = config.supervisor.model
    model = ChatOpenAI(
        model=model_config.name,
        base_url=model_config.base_url,
        api_key=SecretStr(model_config.api_key),
    )
    return create_deep_agent(
        model=model,
        system_prompt="You are a helpful assistant.",
    )
