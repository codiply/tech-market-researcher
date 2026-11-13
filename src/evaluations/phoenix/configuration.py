"""Config models. All classes here are named XxxConfig."""

import os
from pathlib import Path

from pydantic import Field

from tech_market_researcher.core.common import BasePydanticModel
from tech_market_researcher.core.config_loader import ConfigLoader, ConfigLoaderParams
from tech_market_researcher.core.constants import BASE_DEBUG_DIR_PATH, DEBUG_DUMP_RENDERED_CONFIGS_ENV_VAR

CONFIG_DIR = Path("config") / "evaluations" / "phoenix"


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
    """Concurrency settings for the Phoenix evaluation run.

    ``max_concurrent`` is passed as the `concurrency` parameter of the Phoenix async
    experiment runner, which bounds the tasks and evaluations in flight.
    ``None`` or ``0`` leaves the Phoenix default (3) in place.
    """

    # Max concurrent gateway requests (agent invocations and judge calls combined); 0 means no limit.
    max_concurrent: int | None = Field(default=None, ge=0)


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
        debug_dir_path: Path | None = None
        if os.environ.get(DEBUG_DUMP_RENDERED_CONFIGS_ENV_VAR, "").lower() in {"true", "1"}:
            debug_dir_path = BASE_DEBUG_DIR_PATH / "config" / "evaluations" / "phoenix"
        _CONFIG_LOADER = ConfigLoader(
            PhoenixEvaluationConfig,
            CONFIG_DIR,
            ConfigLoaderParams(debug_dir_path=debug_dir_path),
        )
    return _CONFIG_LOADER


# pylint: enable=duplicate-code


def load_config(name: str) -> PhoenixEvaluationConfig:
    """Load the evaluation config at config/evaluations/phoenix/<name>.yaml."""
    return get_config_loader().load(name)
