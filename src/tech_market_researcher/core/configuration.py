"""Config models. All classes here are named XxxConfig."""

import logging
import os
import re
from enum import Enum
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, model_validator

from tech_market_researcher.core.common import BasePydanticModel
from tech_market_researcher.core.config_loader import ConfigLoader, ConfigLoaderParams
from tech_market_researcher.core.constants import (
    AGENT_DEFAULT_LOG_LEVEL,
    AGENT_DEFAULT_LOG_ROTATION_BACKUP_COUNT,
    AGENT_DEFAULT_LOG_ROTATION_MAX_SIZE_MB,
    AGENT_DEFAULT_LOG_ROTATION_WHEN,
    AGENT_DEFAULT_LOGS_DIR,
    ARCHIVE_RENDERED_CONFIGS_ENV_VAR,
    BASE_DEBUG_DIR_PATH,
    DEBUG_CONFIG_ARCHIVE_FOLDER,
    DEBUG_CONFIG_ARCHIVE_MAX_COUNT,
    DEBUG_CONFIG_FOLDER,
    DEBUG_CONFIG_LATEST_FOLDER,
    DEBUG_DUMP_RENDERED_CONFIGS_ENV_VAR,
)
from tech_market_researcher.core.environment import env_flag_is_true

CONFIG_DIR = Path("config") / "agent"
# Loader-specific path under the debug config folder.
DEBUG_CONFIG_LOADER_PATH = Path("agent")


class ModelProviderConfig(BasePydanticModel):
    """Connection settings for a model provider."""

    base_url: str | None = None
    base_url_env_var_name: str | None = None
    api_key_env_var_name: str | None = None

    @model_validator(mode="after")
    def _check_base_url(self) -> "ModelProviderConfig":
        if (self.base_url is None) == (self.base_url_env_var_name is None):
            raise ValueError("model provider requires exactly one of base_url or base_url_env_var_name")
        return self

    @property
    def resolved_base_url(self) -> str:
        """Base URL, taken from the literal value or the env var base_url_env_var_name."""
        if self.base_url is not None:
            return self.base_url
        assert self.base_url_env_var_name is not None  # Validated by _check_base_url
        return os.environ[self.base_url_env_var_name]

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


class WebpageIndexerConfig(BasePydanticModel):
    """Web-page indexer tool settings. Omit the section (or set enabled: false) to disable.

    Fetched pages are chunked into token-bounded chunks before being embedded, so each chunk stays
    within the embeddings model's input limit and remains individually retrievable.
    """

    enabled: bool = False
    collection_name: str | None = None
    embeddings_model: ModelConfig | None = None
    chunk_size_tokens: int = 500
    chunk_overlap_tokens: int = 50

    @model_validator(mode="after")
    def _validate_enabled_fields(self) -> "WebpageIndexerConfig":
        if self.enabled and (self.collection_name is None or self.embeddings_model is None):
            msg = "webpage_indexer requires collection_name and embeddings_model when enabled"
            raise ValueError(msg)
        if not 0 <= self.chunk_overlap_tokens < self.chunk_size_tokens:
            msg = "webpage_indexer chunk_overlap_tokens must be less than chunk_size_tokens"
            raise ValueError(msg)
        return self


class CheckpointerBackend(str, Enum):
    """Available conversation checkpointer backends."""

    MEMORY = "memory"
    POSTGRES = "postgres"


class PostgresCheckpointerConfig(BasePydanticModel):
    """Connection settings for the postgres checkpointer backend."""

    host: str = "localhost"
    port: int = 5435
    db_name: str = "agent"
    db_user: str = "agent"
    db_password_env_var_name: str = "AGENT_DB_PASSWORD"
    # Schema that holds the checkpoint tables; other features can use their own schemas.
    schema_name: str = "checkpoints"

    @property
    def db_password(self) -> SecretStr:
        """Database password from the env var db_password_env_var_name."""
        return SecretStr(os.environ[self.db_password_env_var_name])


class CheckpointerConfig(BasePydanticModel):
    """Conversation checkpointer settings. Omit the section to use the memory backend."""

    backend: CheckpointerBackend = CheckpointerBackend.MEMORY
    postgres: PostgresCheckpointerConfig | None = None

    @model_validator(mode="after")
    def _check_postgres(self) -> "CheckpointerConfig":
        if self.backend is CheckpointerBackend.POSTGRES and self.postgres is None:
            raise ValueError("the postgres section is required when the backend is postgres.")
        return self


class CapabilitiesConfig(BasePydanticModel):
    """Settings for the tools and middlewares available to the agent."""

    runtime_context_debugger: RuntimeContextDebuggerConfig = Field(default_factory=RuntimeContextDebuggerConfig)
    current_datetime: CurrentDateTimeConfig = Field(default_factory=CurrentDateTimeConfig)
    fetch_page: FetchPageConfig = Field(default_factory=FetchPageConfig)
    search: SearchToolConfig = Field(default_factory=SearchToolConfig)
    todo_list: TodoListConfig = Field(default_factory=TodoListConfig)
    webpage_indexer: WebpageIndexerConfig = Field(default_factory=WebpageIndexerConfig)


class SupervisorConfig(BasePydanticModel):
    """Supervisor agent settings."""

    model: ModelConfig
    system_prompt: str
    checkpointer: CheckpointerConfig = Field(default_factory=CheckpointerConfig)
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


class PhoenixObservabilityConfig(BasePydanticModel):
    """Phoenix observability settings. Omit the section (or set enabled: false) to disable."""

    enabled: bool = False
    project_name: str = ""

    @model_validator(mode="after")
    def _check_project_name(self) -> "PhoenixObservabilityConfig":
        if self.enabled and not self.project_name:
            raise ValueError("project_name is required when enabled is true.")
        return self


class ObservabilityConfig(BasePydanticModel):
    """Observability settings. Omit the section to disable both MLflow and Phoenix."""

    mlflow: MlflowObservabilityConfig = Field(default_factory=MlflowObservabilityConfig)
    phoenix: PhoenixObservabilityConfig = Field(default_factory=PhoenixObservabilityConfig)


class IndexingConfig(BasePydanticModel):
    """Vector index (Chroma) connection settings."""

    chroma_url_env_var: str = "CHROMA_URL"
    # What to do when an existing collection was created with a different embeddings model:
    # fail, or re-embed the collection's records with the new model (mirrors
    # EmbeddingModelMismatchBehavior; Literal here to avoid a circular import).
    embedding_model_mismatch_behavior: Literal["raise", "migrate"] = "raise"
    # Suffix every collection name with the conversation's thread id, so each
    # conversation reads from and writes to its own collections.
    do_suffix_collection_name_with_thread_id: bool = False
    # Extra metadata recorded on every collection the agent creates.
    additional_collection_metadata: dict[str, str] = Field(default_factory=dict)


class ToolDefinitionsDebugConfig(BasePydanticModel):
    """Tool-definitions dump settings. Omit the section (or set enabled: false) to disable."""

    enabled: bool = False


class DebugConfig(BasePydanticModel):
    """Debug dumps written under the .debug folder. Omit the section to disable them all."""

    tool_definitions: ToolDefinitionsDebugConfig = Field(default_factory=ToolDefinitionsDebugConfig)


class LoggingConfig(BasePydanticModel):
    """Agent file logging settings."""

    directory: Path = AGENT_DEFAULT_LOGS_DIR
    level: str = AGENT_DEFAULT_LOG_LEVEL
    # Rollover schedule passed to TimedRotatingFileHandler: "midnight"/"D", "H", "M", "S",
    # or "W0"-"W6" (weekly, Monday-Sunday).
    rotation_when: str = AGENT_DEFAULT_LOG_ROTATION_WHEN
    # Number of rotated log files kept alongside the current one.
    rotation_backup_count: int = AGENT_DEFAULT_LOG_ROTATION_BACKUP_COUNT
    # Size cap in megabytes that also triggers a rollover of the current log file.
    rotation_max_size_mb: float = AGENT_DEFAULT_LOG_ROTATION_MAX_SIZE_MB

    @model_validator(mode="after")
    def _check_level(self) -> "LoggingConfig":
        valid = sorted(logging.getLevelNamesMapping())
        if self.level.upper() not in logging.getLevelNamesMapping():
            raise ValueError(f"level must be one of {valid}, got '{self.level}'.")
        return self

    @model_validator(mode="after")
    def _check_rotation_when(self) -> "LoggingConfig":
        if self.rotation_when.upper() not in ("S", "M", "H", "D", "MIDNIGHT") and not re.fullmatch(
            r"W[0-6]", self.rotation_when.upper()
        ):
            raise ValueError(
                f"rotation_when must be one of S, M, H, D, midnight, or W0-W6, got '{self.rotation_when}'."
            )
        return self

    @model_validator(mode="after")
    def _check_rotation_backup_count(self) -> "LoggingConfig":
        if self.rotation_backup_count < 0:
            raise ValueError(f"rotation_backup_count must be >= 0, got {self.rotation_backup_count}.")
        return self

    @model_validator(mode="after")
    def _check_rotation_max_size_mb(self) -> "LoggingConfig":
        if self.rotation_max_size_mb <= 0:
            raise ValueError(f"rotation_max_size_mb must be > 0, got {self.rotation_max_size_mb}.")
        return self


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
        base_url=provider.resolved_base_url,
        api_key=provider.api_key,
    )


class ApplicationConfig(BasePydanticModel):
    """Application settings."""

    environment: str
    model_providers: dict[str, ModelProviderConfig]
    supervisor: SupervisorConfig
    indexing: IndexingConfig = Field(default_factory=IndexingConfig)
    observability: ObservabilityConfig = Field(default_factory=ObservabilityConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    debug: DebugConfig = Field(default_factory=DebugConfig)

    def get_model_connection_details(self, model: ModelConfig) -> ModelConnectionDetails:
        """Resolve a model's provider into its full connection details."""
        return resolve_model_connection_details(self.model_providers, model)


# pylint: disable=duplicate-code  # Same lazy-loader pattern as the other configuration.py files.
_CONFIG_LOADER: ConfigLoader | None = None


def get_config_loader() -> ConfigLoader:
    """Lazily construct the config loader on first use."""
    global _CONFIG_LOADER  # pylint: disable=global-statement
    if _CONFIG_LOADER is None:
        base_dir = BASE_DEBUG_DIR_PATH / DEBUG_CONFIG_FOLDER / DEBUG_CONFIG_LOADER_PATH
        debug_dir_path: Path | None = None
        archive_dir_path: Path | None = None
        if env_flag_is_true(DEBUG_DUMP_RENDERED_CONFIGS_ENV_VAR):
            debug_dir_path = base_dir / DEBUG_CONFIG_LATEST_FOLDER
        if env_flag_is_true(ARCHIVE_RENDERED_CONFIGS_ENV_VAR):
            archive_dir_path = base_dir / DEBUG_CONFIG_ARCHIVE_FOLDER
        _CONFIG_LOADER = ConfigLoader(
            ApplicationConfig,
            CONFIG_DIR,
            ConfigLoaderParams(
                debug_dir_path=debug_dir_path,
                archive_dir_path=archive_dir_path,
                archive_max_count=DEBUG_CONFIG_ARCHIVE_MAX_COUNT,
            ),
        )
    return _CONFIG_LOADER


# pylint: enable=duplicate-code


def load_config(name: str, extra_mixins: list[str] | None = None) -> ApplicationConfig:
    """Load the agent config at config/agent/<name>.yaml, applying the extra mixins on top of it."""
    return get_config_loader().load(name, extra_mixins=extra_mixins)
