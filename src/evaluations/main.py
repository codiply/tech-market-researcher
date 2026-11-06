"""Run evaluations of the tech market researcher agent."""

from dotenv import load_dotenv

load_dotenv()

# These imports must run after load_dotenv(): the agent talks to the MLflow
# AI Gateway and its clients read env vars at import time.
# pylint: disable=wrong-import-position
from evaluations.common.arguments import EvalsCommandLineArguments, get_args
from evaluations.common.constants import EvalsPlatform
from evaluations.mlflow.configuration import load_config as load_evals_config
from evaluations.mlflow.evaluate import mlflow_evaluate
from tech_market_researcher.agents.main import create_agent

# pylint: enable=wrong-import-position


def main() -> None:
    """Entry point for the evaluations CLI."""
    eval_args = get_args(EvalsCommandLineArguments)
    agent = create_agent(eval_args.as_agent_args())
    if eval_args.platform == EvalsPlatform.MLLFLOW:
        config = load_evals_config(eval_args.evals_config_name)
        mlflow_evaluate(config, agent)
    else:
        raise NotImplementedError(f"Platform '{eval_args.platform.value}' is not implemented yet.")


if __name__ == "__main__":
    main()
