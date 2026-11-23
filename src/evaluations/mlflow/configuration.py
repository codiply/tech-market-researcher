"""Config models. All classes here are named XxxConfig."""

from pathlib import Path
from typing import Any

from evaluations.common.constants import DEFAULT_EVALS_CONFIG_NAME
from tech_market_researcher.core.common import BasePydanticModel
from tech_market_researcher.core.config_loader import ConfigLoader, ConfigLoaderParams
from tech_market_researcher.core.constants import (
    ARCHIVE_RENDERED_CONFIGS_ENV_VAR,
    BASE_DEBUG_DIR_PATH,
    DEBUG_CONFIG_ARCHIVE_FOLDER,
    DEBUG_CONFIG_ARCHIVE_MAX_COUNT,
    DEBUG_CONFIG_FOLDER,
    DEBUG_CONFIG_LATEST_FOLDER,
    DEBUG_DUMP_RENDERED_CONFIGS_ENV_VAR,
)
from tech_market_researcher.core.environment import env_flag_is_true

CONFIG_DIR = Path("config") / "evaluations" / "mlflow"
# Loader-specific path under the debug config folder.
DEBUG_CONFIG_LOADER_PATH = Path("evaluations") / "mlflow"


class BaseScorerConfig(BasePydanticModel):
    """Settings for a single scorer."""

    enabled: bool = False
    # Override the judge model for this scorer; defaults to the global judge model.
    model_name_override: str | None = None


class ScorerWithInferenceParamsConfig(BaseScorerConfig):
    """Settings for scorers that accept judge sampling parameters."""

    # Sampling parameters forwarded to the judge model, e.g. {temperature: 0.0}.
    inference_params: dict[str, Any] | None = None


class ToolCallCorrectnessScorerConfig(BaseScorerConfig):
    """Settings for the `tool_call_correctness` scorer."""

    # Compare tool calls exactly against expectations instead of using the judge model.
    should_exact_match: bool = False
    # Enforce the expected order of tool calls (only relevant with should_exact_match).
    should_consider_ordering: bool = False


class GuidelinesInstanceConfig(BasePydanticModel):
    """A single instance of the `guidelines` scorer."""

    name: str
    guidelines: list[str]


class GuidelinesScorerConfig(BaseScorerConfig):
    """Settings for the `guidelines` scorer, which applies global guidelines."""

    instances: list[GuidelinesInstanceConfig] = []


class ScorersConfig(BasePydanticModel):
    """Which scorers to run, keyed by scorer name."""

    completeness: BaseScorerConfig
    correctness: BaseScorerConfig
    equivalence: ScorerWithInferenceParamsConfig
    expectations_guidelines: BaseScorerConfig
    fluency: BaseScorerConfig
    guidelines: GuidelinesScorerConfig
    relevance_to_query: BaseScorerConfig
    retrieval_groundedness: BaseScorerConfig
    retrieval_relevance: ScorerWithInferenceParamsConfig
    retrieval_sufficiency: BaseScorerConfig
    safety: BaseScorerConfig
    source_document_groundedness: BaseScorerConfig
    summarization: BaseScorerConfig
    tool_call_correctness: ToolCallCorrectnessScorerConfig
    tool_call_efficiency: BaseScorerConfig


class HarnessConfig(BasePydanticModel):
    """Concurrency and rate-limit settings for the MLflow evaluation harness.

    Values are applied as MLFLOW_GENAI_EVAL_* environment variables before the run
    starts. ``None`` leaves the variable unset so the MLflow default applies.
    """

    # Max concurrent agent (predict_fn) invocations.
    max_workers: int | None = None
    # Max concurrent scorers per record.
    max_scorer_workers: int | None = None
    # Max agent invocations per second; 0 disables rate limiting.
    predict_rate_limit: int | float | None = None
    # Max scorer (judge) calls per second; 0 disables. Default is predict_rate x num_scorers.
    scorer_rate_limit: int | float | None = None
    # Retries on 429 errors for both predict_fn and scorer calls; 0 disables.
    max_retries: int | None = None


class MlflowEvaluationConfig(BasePydanticModel):
    """Evaluation settings. The judge runs against an endpoint of the MLflow AI Gateway."""

    experiment_name: str
    dataset_name: str
    judge_model_name: str
    scorers: ScorersConfig
    harness: HarnessConfig = HarnessConfig()


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
            MlflowEvaluationConfig,
            CONFIG_DIR,
            ConfigLoaderParams(
                default_config_name=DEFAULT_EVALS_CONFIG_NAME,
                debug_dir_path=debug_dir_path,
                archive_dir_path=archive_dir_path,
                archive_max_count=DEBUG_CONFIG_ARCHIVE_MAX_COUNT,
            ),
        )
    return _CONFIG_LOADER


# pylint: enable=duplicate-code


def load_config(name: str, extra_mixins: list[str] | None = None) -> MlflowEvaluationConfig:
    """Load the evaluation config at config/evaluations/mlflow/<name>.yaml, applying the extra mixins on top of it."""
    return get_config_loader().load(name, extra_mixins=extra_mixins)
