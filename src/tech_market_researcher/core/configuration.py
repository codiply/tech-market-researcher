"""Config models. All classes here are named XxxConfig."""

import os
from enum import Enum
from pathlib import Path

from pydantic import Field, SecretStr, model_validator

from tech_market_researcher.core.common import BasePydanticModel
from tech_market_researcher.core.config_loader import ConfigLoader

CONFIG_DIR = Path("config") / "agent"


class ModelProviderConfig(BasePydanticModel):
    """Connection settings for a model provider."""

    base_url: str
    api_key_env_var_name: str | None = None

    @property
    def api_key(self) -> SecretStr:
        """API key from the env var api_key_env_var_name, or dummy if unset."""
        if self.api_key_env_var_name:
            return SecretStr(os.environ[self.api_key_env_var_name])
        return SecretStr("dummy")


class ModelConfig(BasePydanticModel):
    """Model settings."""

    name: str
    provider: str


class ModelConnectionDetails(BasePydanticModel):
    """Resolved connection details for one model."""

    model_name: str
    base_url: str
    api_key: SecretStr


class SearchProvider(str, Enum):
    """Available search tool providers."""

    TAVILY = "tavily"
    SEARXNG = "searxng"


class SearchToolConfig(BasePydanticModel):
    """Search tool settings. Omit the section (or set enabled: false) to disable."""

    enabled: bool = False
    provider: SearchProvider = SearchProvider.SEARXNG


class FetchPageConfig(BasePydanticModel):
    """Fetch-page tool settings. Omit the section (or set enabled: false) to disable."""

    enabled: bool = False


class TodoListConfig(BasePydanticModel):
    """Todo-list middleware settings."""

    enabled: bool = False


class RuntimeContextDebuggerConfig(BasePydanticModel):
    """Runtime-context debugger tool settings. Omit the section (or set enabled: false) to disable."""

    enabled: bool = False


class CurrentDateTimeConfig(BasePydanticModel):
    """Current-datetime tool settings. Omit the section (or set enabled: false) to disable."""

    enabled: bool = False


class CapabilitiesConfig(BasePydanticModel):
    """Settings for the tools and middlewares available to the agent."""

    runtime_context_debugger: RuntimeContextDebuggerConfig = Field(default_factory=RuntimeContextDebuggerConfig)
    current_datetime: CurrentDateTimeConfig = Field(default_factory=CurrentDateTimeConfig)
    fetch_page: FetchPageConfig = Field(default_factory=FetchPageConfig)
    search: SearchToolConfig = Field(default_factory=SearchToolConfig)
    todo_list: TodoListConfig = Field(default_factory=TodoListConfig)


class SupervisorConfig(BasePydanticModel):
    """Supervisor agent settings."""

    model: ModelConfig
    system_prompt: str
    capabilities: CapabilitiesConfig


class MlflowObservabilityConfig(BasePydanticModel):
    """MLflow observability settings. Omit the section (or set enabled: false) to disable."""

    enabled: bool = False
    experiment_name: str = ""

    @model_validator(mode="after")
    def _check_experiment_name(self) -> "MlflowObservabilityConfig":
        if self.enabled and not self.experiment_name:
            raise ValueError("experiment_name is required when enabled is true.")
        return self


class ArizeObservabilityConfig(BasePydanticModel):
    """Arize observability settings. Omit the section (or set enabled: false) to disable."""

    enabled: bool = False
    project_name: str = ""

    @model_validator(mode="after")
    def _check_project_name(self) -> "ArizeObservabilityConfig":
        if self.enabled and not self.project_name:
            raise ValueError("project_name is required when enabled is true.")
        return self


class ObservabilityConfig(BasePydanticModel):
    """Observability settings. Omit the section to disable both MLflow and Arize."""

    mlflow: MlflowObservabilityConfig = Field(default_factory=MlflowObservabilityConfig)
    arize: ArizeObservabilityConfig = Field(default_factory=ArizeObservabilityConfig)


def resolve_model_connection_details(
    model_providers: dict[str, ModelProviderConfig], model: ModelConfig
) -> ModelConnectionDetails:
    """Resolve a model's provider into its full connection details."""
    provider = model_providers.get(model.provider)
    if provider is None:
        available = ", ".join(sorted(model_providers))
        raise ValueError(f"Unknown model provider '{model.provider}'. Available: {available}.")
    return ModelConnectionDetails(
        model_name=model.name,
        base_url=provider.base_url,
        api_key=provider.api_key,
    )


class ApplicationConfig(BasePydanticModel):
    """Application settings."""

    environment: str
    model_providers: dict[str, ModelProviderConfig]
    supervisor: SupervisorConfig
    observability: ObservabilityConfig = Field(default_factory=ObservabilityConfig)

    def get_model_connection_details(self, model: ModelConfig) -> ModelConnectionDetails:
        """Resolve a model's provider into its full connection details."""
        return resolve_model_connection_details(self.model_providers, model)


CONFIG_LOADER = ConfigLoader(ApplicationConfig, CONFIG_DIR)


def load_config(name: str) -> ApplicationConfig:
    """Load the agent config at config/agent/<name>.yaml."""
    return CONFIG_LOADER.load(name)
