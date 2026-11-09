"""Config models. All classes here are named XxxConfig."""

import os
from pathlib import Path

from tech_market_researcher.core.common import BasePydanticModel
from tech_market_researcher.core.config_loader import ConfigLoader, ConfigLoaderParams
from tech_market_researcher.core.constants import BASE_DEBUG_DIR_PATH, DEBUG_DUMP_RENDERED_CONFIGS_ENV_VAR

CONFIG_DIR = Path("config") / "evaluations" / "arize"


class BaseEvaluatorConfig(BasePydanticModel):
    """Settings for a single evaluator."""

    enabled: bool = False
    # Override the judge model for this evaluator; defaults to the global judge model.
    model_name_override: str | None = None


class EvaluatorsConfig(BasePydanticModel):
    """Which evaluators to run, keyed by evaluator name."""

    conciseness: BaseEvaluatorConfig
    correctness: BaseEvaluatorConfig
    hallucination: BaseEvaluatorConfig
    refusal: BaseEvaluatorConfig


class ArizeEvaluationConfig(BasePydanticModel):
    """Evaluation settings. The judge runs against an endpoint of the MLflow AI Gateway."""

    dataset_name: str
    judge_model_name: str
    evaluators: EvaluatorsConfig


# pylint: disable=duplicate-code  # Same lazy-loader pattern as the other configuration.py files.
_CONFIG_LOADER: ConfigLoader | None = None


def get_config_loader() -> ConfigLoader:
    """Lazily construct the config loader on first use."""
    global _CONFIG_LOADER  # pylint: disable=global-statement
    if _CONFIG_LOADER is None:
        debug_dir_path: Path | None = None
        if os.environ.get(DEBUG_DUMP_RENDERED_CONFIGS_ENV_VAR, "").lower() in {"true", "1"}:
            debug_dir_path = BASE_DEBUG_DIR_PATH / "config" / "evaluations" / "arize"
        _CONFIG_LOADER = ConfigLoader(
            ArizeEvaluationConfig,
            CONFIG_DIR,
            ConfigLoaderParams(debug_dir_path=debug_dir_path),
        )
    return _CONFIG_LOADER


# pylint: enable=duplicate-code


def load_config(name: str) -> ArizeEvaluationConfig:
    """Load the evaluation config at config/evaluations/arize/<name>.yaml."""
    return get_config_loader().load(name)
