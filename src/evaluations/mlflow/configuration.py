"""Config models. All classes here are named XxxConfig."""

import os
from pathlib import Path
from typing import Any

from tech_market_researcher.core.common import BasePydanticModel
from tech_market_researcher.core.config_loader import ConfigLoader, ConfigLoaderParams
from tech_market_researcher.core.constants import BASE_DEBUG_DIR_PATH, DEBUG_DUMP_RENDERED_CONFIGS_ENV_VAR

CONFIG_DIR = Path("config") / "evaluations" / "mlflow"


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
    summarization: BaseScorerConfig
    tool_call_correctness: ToolCallCorrectnessScorerConfig
    tool_call_efficiency: BaseScorerConfig


class MlflowEvaluationConfig(BasePydanticModel):
    """Evaluation settings. The judge runs against an endpoint of the MLflow AI Gateway."""

    experiment_name: str
    dataset_name: str
    judge_model_name: str
    scorers: ScorersConfig


# pylint: disable=duplicate-code  # Same lazy-loader pattern as the other configuration.py files.
_CONFIG_LOADER: ConfigLoader | None = None


def get_config_loader() -> ConfigLoader:
    """Lazily construct the config loader on first use."""
    global _CONFIG_LOADER  # pylint: disable=global-statement
    if _CONFIG_LOADER is None:
        debug_dir_path: Path | None = None
        if os.environ.get(DEBUG_DUMP_RENDERED_CONFIGS_ENV_VAR, "").lower() in {"true", "1"}:
            debug_dir_path = BASE_DEBUG_DIR_PATH / "config" / "evaluations" / "mlflow"
        _CONFIG_LOADER = ConfigLoader(
            MlflowEvaluationConfig,
            CONFIG_DIR,
            ConfigLoaderParams(debug_dir_path=debug_dir_path),
        )
    return _CONFIG_LOADER


# pylint: enable=duplicate-code


def load_config(name: str) -> MlflowEvaluationConfig:
    """Load the evaluation config at config/evaluations/mlflow/<name>.yaml."""
    return get_config_loader().load(name)
