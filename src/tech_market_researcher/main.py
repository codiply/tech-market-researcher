import os
from importlib.metadata import version

from dotenv import load_dotenv

# Load .env from the current directory (real env vars take precedence)
load_dotenv()

from langchain_openai import ChatOpenAI

from deepagents import create_deep_agent
from rich.console import Console
import mlflow

TRACKING_URI_ENV = "MLFLOW_TRACKING_URI"
EXPERIMENT_ENV = "MLFLOW_EXPERIMENT"
REQUIRED_ENVS = [TRACKING_URI_ENV, EXPERIMENT_ENV]
MODEL = "Qwen3.8-27B-oQ4e-mtp"
# The agent talks to the MLflow AI Gateway, which proxies to the oMLX server.
# The endpoint name must match what scripts/bootstrap.py created.
GATEWAY_ENDPOINT = "oMLX"
__version__ = version("tech-market-researcher")

# Enable auto-tracing for LangChain (and LangGraph-based frameworks like Deep Agent)
mlflow.langchain.autolog()

console = Console()

def main() -> None:
    missing = [name for name in REQUIRED_ENVS if not os.environ.get(name)]
    if missing:
        print(f"Set {', '.join(missing)} to run the agent.")
        return

    tracking_uri = os.environ[TRACKING_URI_ENV].rstrip("/")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(os.environ[EXPERIMENT_ENV])
    mlflow.set_active_model(name=f"tech-market-researcher-v{__version__}".replace(".", "-"))

    llm = ChatOpenAI(
        model=MODEL,
        base_url=f"{tracking_uri}/gateway/proxy/{GATEWAY_ENDPOINT}/v1",
        # The gateway does not validate API keys; a dummy value is fine.
        api_key="dummy",
    )

    agent = create_deep_agent(
        model=llm,
        system_prompt="You are a helpful assistant.",
    )

    print("Chat with the agent. Type 'exit' to quit.")
    messages: list = []
    while True:
        user_input = console.input("[bold red]you> [/bold red]").strip()
        if user_input.lower() in {"exit", "quit"}:
            break
        if not user_input:
            continue

        messages.append({"role": "user", "content": user_input})
        result = agent.invoke({"messages": messages})
        # Keep the full conversation (incl. the agent's reply) for the next turn
        messages = result["messages"]
        console.print("[bold cyan]agent>[/bold cyan]", result['messages'][-1].content)


if __name__ == "__main__":
    main()
