"""Config models. All classes here are named XxxConfig."""

import os
from pathlib import Path

import yaml
from pydantic import BaseModel, SecretStr

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


class SupervisorConfig(BaseModel):
    """Supervisor agent settings."""

    model: ModelConfig


class ApplicationConfig(BaseModel):
    """Application settings."""

    model_providers: dict[str, ModelProviderConfig]
    supervisor: SupervisorConfig

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
