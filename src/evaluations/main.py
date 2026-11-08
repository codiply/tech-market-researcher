"""Run evaluations of the tech market researcher agent."""

from dotenv import load_dotenv

load_dotenv()

# These imports must run after load_dotenv(): the agent talks to the MLflow
# AI Gateway and its clients read env vars at import time.
# pylint: disable=wrong-import-position
from evaluations.arize.configuration import load_config as load_arize_config
from evaluations.arize.evaluate import arize_evaluate
from evaluations.common.arguments import EvalsCommandLineArguments, get_args
from evaluations.common.constants import EVALS_ENVIRONMENT, EvalsPlatform
from evaluations.mlflow.configuration import load_config as load_mlflow_config
from evaluations.mlflow.evaluate import mlflow_evaluate
from tech_market_researcher.agents.main import create_agent
from tech_market_researcher.core.configuration import ApplicationConfig

# pylint: enable=wrong-import-position


def _update_config(config: ApplicationConfig) -> None:
    """Apply evals-specific mutations to the agent config in place."""
    # Evals run as their own environment so their traces can be told apart.
    config.environment = EVALS_ENVIRONMENT


def main() -> None:
    """Entry point for the evaluations CLI."""
    eval_args = get_args(EvalsCommandLineArguments)
    agent = create_agent(eval_args.as_agent_args(), config_mutations_hook=_update_config)
    if eval_args.platform == EvalsPlatform.MLLFLOW:
        mlflow_config = load_mlflow_config(eval_args.evals_config_name)
        mlflow_evaluate(mlflow_config, agent)
    elif eval_args.platform == EvalsPlatform.ARIZE:
        arize_config = load_arize_config(eval_args.evals_config_name)
        arize_evaluate(arize_config, agent)
    else:
        raise NotImplementedError(f"Platform '{eval_args.platform.value}' is not implemented yet.")


if __name__ == "__main__":
    main()
