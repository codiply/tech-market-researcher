"""Config models. All classes here are named XxxConfig."""

import os
import re
from enum import Enum
from pathlib import Path

import yaml
from pydantic import BaseModel, field_validator, model_validator

CONFIG_DIR = Path("config") / "bootstrapping"


class LlmProvider(str, Enum):
    """Gateway provider ids accepted by the MLflow API."""

    OPENAI = "openai"
    BEDROCK = "bedrock"  # shown as "Amazon Bedrock" in the MLflow UI


class EndpointConfig(BaseModel):
    """One gateway endpoint, each serving a single model."""

    name: str
    model_name: str
    api_key_name: str

    @field_validator("name")
    @classmethod
    def _check_name_charset(cls, name: str) -> str:
        """MLflow endpoint names allow only letters, numbers, _, -, and ."""
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", name):
            raise ValueError(
                f"Endpoint name '{name}' may only contain letters, numbers, underscores, hyphens, and dots."
            )
        return name


class LlmConnectionConfig(BaseModel):
    """One LLM connection exposed through the MLflow gateway."""

    api_key_name: str
    provider: LlmProvider
    region_name: str | None = None
    api_key_env_var_name: str | None = None
    api_base_url_env_var_name: str | None = None

    @model_validator(mode="after")
    def _check_region_name(self) -> "LlmConnectionConfig":
        """region_name must be set exactly for bedrock connections."""
        if self.provider is LlmProvider.BEDROCK and self.region_name is None:
            raise ValueError(f"LLM connection '{self.api_key_name}': region_name is required for the bedrock provider.")
        if self.provider is not LlmProvider.BEDROCK and self.region_name is not None:
            raise ValueError(
                f"LLM connection '{self.api_key_name}': region_name is only supported for the bedrock provider."
            )
        return self

    @property
    def api_key(self) -> str:
        """API key from the env var api_key_env_var_name, or dummy if unset."""
        if self.api_key_env_var_name:
            return os.environ[self.api_key_env_var_name]
        return "dummy"

    @property
    def api_base_url(self) -> str | None:
        """Base URL from the env var api_base_url_env_var_name, or None if unset."""
        if self.api_base_url_env_var_name:
            return os.environ.get(self.api_base_url_env_var_name)
        return None


class MlflowConfig(BaseModel):
    """MLflow AI Gateway settings."""

    llm_connections: list[LlmConnectionConfig]
    endpoints: list[EndpointConfig] = []

    @model_validator(mode="after")
    def _check_duplicate_names(self) -> "MlflowConfig":
        """Reject duplicate names: they would be ambiguous references."""
        connections = {c.api_key_name for c in self.llm_connections}
        if len(connections) != len(self.llm_connections):
            raise ValueError("Duplicate api_key_name in llm_connections.")
        endpoints = {e.name for e in self.endpoints}
        if len(endpoints) != len(self.endpoints):
            raise ValueError("Duplicate name in endpoints.")
        return self

    @property
    def connections_by_key_name(self) -> dict[str, LlmConnectionConfig]:
        """Connections indexed by api_key_name."""
        return {connection.api_key_name: connection for connection in self.llm_connections}


class S3Config(BaseModel):
    """S3 (MinIO) artifact storage settings."""

    buckets_to_create: list[str]


class BootstrapConfig(BaseModel):
    """Application settings."""

    mlflow: MlflowConfig
    s3: S3Config


def load_config(name: str) -> BootstrapConfig:
    """Load the bootstrap config at config/bootstrapping/<name>.yaml."""
    data = yaml.safe_load((CONFIG_DIR / f"{name}.yaml").read_text())
    return BootstrapConfig.model_validate(data)
