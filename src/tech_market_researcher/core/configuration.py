"""Config models. All classes here are named XxxConfig."""

import os
from enum import Enum
from pathlib import Path

import yaml
from pydantic import BaseModel, Field, SecretStr, model_validator

CONFIG_DIR = Path("config") / "agent"


class ModelProviderConfig(BaseModel):
    """Connection settings for a model provider."""

    base_url: str
    api_key_env_var_name: str | None = None

    @property
    def api_key(self) -> str:
        """API key from the env var api_key_env_var_name, or dummy if unset."""
        if self.api_key_env_var_name:
            return os.environ[self.api_key_env_var_name]
        return "dummy"


class ModelConfig(BaseModel):
    """Model settings."""

    name: str
    provider: str


class ModelConnectionDetails(BaseModel):
    """Resolved connection details for one model."""

    model_name: str
    base_url: str
    api_key: SecretStr


class SearchProvider(str, Enum):
    """Available search tool providers."""

    TAVILY = "tavily"
    SEARXNG = "searxng"


class SearchToolConfig(BaseModel):
    """Search tool settings. Omit the section (or set enabled: false) to disable."""

    enabled: bool = False
    provider: SearchProvider = SearchProvider.SEARXNG


class FetchPageConfig(BaseModel):
    """Fetch-page tool settings. Omit the section (or set enabled: false) to disable."""

    enabled: bool = False


class TodoListConfig(BaseModel):
    """Todo-list middleware settings."""

    enabled: bool = False


class RuntimeContextDebuggerConfig(BaseModel):
    """Runtime-context debugger tool settings. Omit the section (or set enabled: false) to disable."""

    enabled: bool = False


class CurrentDateTimeConfig(BaseModel):
    """Current-datetime tool settings. Omit the section (or set enabled: false) to disable."""

    enabled: bool = False


class CapabilitiesConfig(BaseModel):
    """Settings for the tools and middlewares available to the agent."""

    runtime_context_debugger: RuntimeContextDebuggerConfig = Field(default_factory=RuntimeContextDebuggerConfig)
    current_datetime: CurrentDateTimeConfig = Field(default_factory=CurrentDateTimeConfig)
    fetch_page: FetchPageConfig = Field(default_factory=FetchPageConfig)
    search: SearchToolConfig = Field(default_factory=SearchToolConfig)
    todo_list: TodoListConfig = Field(default_factory=TodoListConfig)


class SupervisorConfig(BaseModel):
    """Supervisor agent settings."""

    model: ModelConfig
    capabilities: CapabilitiesConfig


class MlflowObservabilityConfig(BaseModel):
    """MLflow observability settings. Omit the section (or set enabled: false) to disable."""

    enabled: bool = False
    experiment_template: str = ""

    @model_validator(mode="after")
    def _check_experiment_template(self) -> "MlflowObservabilityConfig":
        if self.enabled and not self.experiment_template:
            raise ValueError("experiment_template is required when enabled is true.")
        return self


class ArizeObservabilityConfig(BaseModel):
    """Arize observability settings. Omit the section (or set enabled: false) to disable."""

    enabled: bool = False
    project_name: str = ""

    @model_validator(mode="after")
    def _check_project_name(self) -> "ArizeObservabilityConfig":
        if self.enabled and not self.project_name:
            raise ValueError("project_name is required when enabled is true.")
        return self


class ObservabilityConfig(BaseModel):
    """Observability settings. Omit the section to disable both MLflow and Arize."""

    mlflow: MlflowObservabilityConfig = Field(default_factory=MlflowObservabilityConfig)
    arize: ArizeObservabilityConfig = Field(default_factory=ArizeObservabilityConfig)


class ApplicationConfig(BaseModel):
    """Application settings."""

    model_providers: dict[str, ModelProviderConfig]
    supervisor: SupervisorConfig
    observability: ObservabilityConfig = Field(default_factory=ObservabilityConfig)

    def get_model_connection_details(self, model: ModelConfig) -> ModelConnectionDetails:
        """Resolve a model's provider into its full connection details."""
        provider = self.model_providers.get(model.provider)
        if provider is None:
            available = ", ".join(sorted(self.model_providers))
            raise ValueError(f"Unknown model provider '{model.provider}'. Available: {available}.")
        return ModelConnectionDetails(
            model_name=model.name,
            base_url=provider.base_url,
            api_key=SecretStr(provider.api_key),
        )


def load_config(name: str) -> ApplicationConfig:
    """Load the agent config at config/agent/<name>.yaml."""
    data = yaml.safe_load((CONFIG_DIR / f"{name}.yaml").read_text())
    return ApplicationConfig.model_validate(data)
