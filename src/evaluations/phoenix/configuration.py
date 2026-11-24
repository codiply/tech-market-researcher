"""Config models. All classes here are named XxxConfig."""

from pathlib import Path

from pydantic import Field

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

CONFIG_DIR = Path("config") / "evaluations" / "phoenix"
# Loader-specific path under the debug config folder.
DEBUG_CONFIG_LOADER_PATH = Path("evaluations") / "phoenix"


class BaseEvaluatorConfig(BasePydanticModel):
    """Settings for a single evaluator."""

    enabled: bool = False
    # Override the judge model for this evaluator; defaults to the global judge model.
    model_name_override: str | None = None


class MatchesRegexConfig(BaseEvaluatorConfig):
    """`matches_regex` needs a regex `pattern` to match the agent's answer against."""

    pattern: str = ""


class EvaluatorsConfig(BasePydanticModel):
    """Which evaluators to run, keyed by evaluator name."""

    completeness: BaseEvaluatorConfig
    conciseness: BaseEvaluatorConfig
    correctness: BaseEvaluatorConfig
    expected_facts: BaseEvaluatorConfig
    exact_match: BaseEvaluatorConfig
    faithfulness: BaseEvaluatorConfig
    guidelines: BaseEvaluatorConfig
    hallucination: BaseEvaluatorConfig
    matches_regex: MatchesRegexConfig
    pii_detection: BaseEvaluatorConfig
    refusal: BaseEvaluatorConfig
    toxicity: BaseEvaluatorConfig
    user_friction: BaseEvaluatorConfig


class HarnessConfig(BasePydanticModel):
    """Concurrency and timeout settings for the Phoenix evaluation run.

    ``max_concurrent`` is passed as the `concurrency` parameter of the Phoenix async
    experiment runner, which bounds the tasks and evaluations in flight.
    ``None`` or ``0`` leaves the Phoenix default (3) in place.
    ``task_timeout_seconds`` is the per-task execution timeout in seconds; ``None`` leaves the
    Phoenix default (60 s), which is too short for a full agent turn with tools.
    """

    # Max concurrent gateway requests (agent invocations and judge calls combined); 0 means no limit.
    max_concurrent: int | None = Field(default=None, ge=0)
    # Per-task execution timeout in seconds (agent invocations and judge calls).
    task_timeout_seconds: int | None = Field(default=None, gt=0)


class PhoenixEvaluationConfig(BasePydanticModel):
    """Evaluation settings. The judge runs against an endpoint of the MLflow AI Gateway."""

    dataset_name: str
    judge_model_name: str
    evaluators: EvaluatorsConfig
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
            PhoenixEvaluationConfig,
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


def load_config(name: str, extra_mixins: list[str] | None = None) -> PhoenixEvaluationConfig:
    """Load the evaluation config at config/evaluations/phoenix/<name>.yaml, applying the extra mixins on top of it."""
    return get_config_loader().load(name, extra_mixins=extra_mixins)
