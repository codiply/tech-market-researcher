"""Config models. All classes here are named XxxConfig."""

import os
from pathlib import Path

import yaml
from pydantic import BaseModel

CONFIG_DIR = Path("config") / "researcher"


class ModelConfig(BaseModel):
    """Model settings."""

    name: str
    base_url: str
    api_key_env_var_name: str | None = None

    @property
    def api_key(self) -> str:
        """API key from the env var api_key_env_var_name, or dummy if unset."""
        if self.api_key_env_var_name:
            return os.environ[self.api_key_env_var_name]
        return "dummy"


class SupervisorConfig(BaseModel):
    """Supervisor agent settings."""

    model: ModelConfig


class ApplicationConfig(BaseModel):
    """Application settings."""

    supervisor: SupervisorConfig


def load_config(name: str) -> ApplicationConfig:
    """Load the researcher config at config/researcher/<name>.yaml."""
    data = yaml.safe_load((CONFIG_DIR / f"{name}.yaml").read_text())
    return ApplicationConfig.model_validate(data)
