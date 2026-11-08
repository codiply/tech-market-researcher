"""Config models. All classes here are named XxxConfig."""

import os
import re
from enum import Enum
from pathlib import Path

import yaml
from pydantic import Field, SecretStr, field_validator, model_validator

from tech_market_researcher.core.common import BasePydanticModel

CONFIG_DIR = Path("config") / "bootstrapping"


class LlmProvider(str, Enum):
    """Gateway provider ids accepted by the MLflow API."""

    OPENAI = "openai"
    BEDROCK = "bedrock"  # shown as "Amazon Bedrock" in the MLflow UI


class EndpointConfig(BasePydanticModel):
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


class LlmConnectionConfig(BasePydanticModel):
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
    def api_key(self) -> SecretStr:
        """API key from the env var api_key_env_var_name, or dummy if unset."""
        if self.api_key_env_var_name:
            return SecretStr(os.environ[self.api_key_env_var_name])
        return SecretStr("dummy")

    @property
    def api_base_url(self) -> str | None:
        """Base URL from the env var api_base_url_env_var_name, or None if unset."""
        if self.api_base_url_env_var_name:
            return os.environ.get(self.api_base_url_env_var_name)
        return None


class BudgetPolicyResetPeriod(str, Enum):
    """Budget policy window, as written in the config file."""

    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"

    @property
    def duration_unit(self) -> str:
        """MLflow BudgetDurationUnit matching this period (always value 1)."""
        return {
            BudgetPolicyResetPeriod.DAILY: "DAYS",
            BudgetPolicyResetPeriod.WEEKLY: "WEEKS",
            BudgetPolicyResetPeriod.MONTHLY: "MONTHS",
        }[self]


class BudgetPolicyAction(str, Enum):
    """Gateway action when a budget policy is exceeded (alert is not exposed for now)."""

    REJECT = "reject"

    @property
    def gateway_action(self) -> str:
        """MLflow BudgetAction matching this action."""
        if self is BudgetPolicyAction.REJECT:
            return "REJECT"
        raise NotImplementedError(f"No gateway action mapping for {self!r}.")


class BudgetPolicyConfig(BasePydanticModel):
    """One global AI gateway budget policy."""

    amount_usd: float = Field(gt=0)
    reset_period: BudgetPolicyResetPeriod
    action: BudgetPolicyAction


class DatasetFiltersConfig(BasePydanticModel):
    """Record predicates for cherry-picking a subset of a dataset's folders.

    Each field is optional: None matches both cases, True selects only records
    with the attribute, False only records without it.
    """

    has_expected_response: bool | None = None
    has_expected_facts: bool | None = None
    has_guidelines: bool | None = None


class DatasetSourceConfig(BasePydanticModel):
    """A dataset sourced from YAML files in local folders under assets/datasets/."""

    dataset_name: str
    local_directories: list[str] = []
    filters: DatasetFiltersConfig = Field(default_factory=DatasetFiltersConfig)


class MlflowDatasetConfig(DatasetSourceConfig):
    """One evaluation dataset provisioned under an MLflow experiment."""

    experiment_name: str


class ArizeDatasetConfig(DatasetSourceConfig):
    """One dataset provisioned in Arize (Phoenix)."""

    description: str | None = None


class ArizeProviderSdk(str, Enum):
    """LLM SDKs Arize (Phoenix) can use to talk to a provider."""

    OPENAI = "openai"


class ArizeApiType(str, Enum):
    """OpenAI API surface the provider serves, as written in the config file."""

    CHAT_COMPLETIONS = "chat_completions"
    RESPONSES = "responses"

    @property
    def graphql_value(self) -> str:
        """Enum value expected by the Phoenix GraphQL API (uppercase)."""
        return self.value.upper()


class ArizeOpenAISettings(BasePydanticModel):
    """OpenAI-SDK settings for one Arize custom provider."""

    api_type: ArizeApiType
    api_key_env_var_name: str | None = None
    api_base_url_env_var_name: str | None = None

    @property
    def api_key(self) -> SecretStr:
        """API key from the env var api_key_env_var_name, or dummy if unset."""
        if self.api_key_env_var_name:
            return SecretStr(os.environ[self.api_key_env_var_name])
        return SecretStr("dummy")

    @property
    def api_base_url(self) -> str | None:
        """Base URL from the env var api_base_url_env_var_name, or None if unset."""
        if self.api_base_url_env_var_name:
            return os.environ.get(self.api_base_url_env_var_name)
        return None


class ArizeAIProviderConfig(BasePydanticModel):
    """One custom LLM provider provisioned in Arize (Phoenix)."""

    provider_name: str
    description: str | None = None
    sdk: ArizeProviderSdk
    provider_string: str
    sdk_specific_settings: ArizeOpenAISettings


class ArizeModelCosts(BasePydanticModel):
    """Token prices for one Arize model, in USD per million tokens.

    Phoenix requires input and output prices; when omitted they are reported as
    zero. Omitted optional kinds are not reported to Phoenix at all.
    """

    input: float = 0.0
    output: float = 0.0
    cache_read: float | None = None
    cache_write: float | None = None
    reasoning: float | None = None


class ArizeModelConfig(BasePydanticModel):
    """One generative model provisioned in Arize for cost tracking."""

    model_name: str
    name_pattern: str
    provider: str | None = None
    costs: ArizeModelCosts = Field(default_factory=ArizeModelCosts)

    @field_validator("name_pattern")
    @classmethod
    def _check_name_pattern(cls, pattern: str) -> str:
        """Phoenix compiles it as a regex to match model ids in spans."""
        re.compile(pattern)
        return pattern


class ArizeConfig(BasePydanticModel):
    """Arize (Phoenix) settings."""

    datasets: list[ArizeDatasetConfig] = []
    ai_providers: list[ArizeAIProviderConfig] = []
    models: list[ArizeModelConfig] = []


class MlflowConfig(BasePydanticModel):
    """MLflow AI Gateway settings."""

    llm_connections: list[LlmConnectionConfig]
    endpoints: list[EndpointConfig] = []
    budget_policies: list[BudgetPolicyConfig] = []
    datasets: list[MlflowDatasetConfig] = []

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


class S3Config(BasePydanticModel):
    """S3 (MinIO) artifact storage settings."""

    buckets_to_create: list[str]


class DebuggingConfig(BasePydanticModel):
    """Debugging aids; disabled by default."""

    # Dump rendered Jinja dataset templates into .debug/datasets/<dir>/<file>.yaml.
    dump_rendered_datasets: bool = False


class BootstrapConfig(BasePydanticModel):
    """Application settings."""

    mlflow: MlflowConfig
    s3: S3Config
    arize: ArizeConfig
    debugging: DebuggingConfig = DebuggingConfig()


def load_config(name: str) -> BootstrapConfig:
    """Load the bootstrap config at config/bootstrapping/<name>.yaml."""
    data = yaml.safe_load((CONFIG_DIR / f"{name}.yaml").read_text())
    return BootstrapConfig.model_validate(data)
