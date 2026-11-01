import os
from importlib.metadata import version

from dotenv import load_dotenv

# Load .env from the current directory (real env vars take precedence)
load_dotenv()

from langchain_openai import ChatOpenAI

from deepagents import create_deep_agent
from rich.console import Console
import mlflow

BASE_URL_ENV = "OPENAI_BASE_URL"
API_KEY_ENV = "OPENAI_API_KEY"
TRACKING_URI_ENV = "MLFLOW_TRACKING_URI"
EXPERIMENT_ENV = "MLFLOW_EXPERIMENT"
REQUIRED_ENVS = [BASE_URL_ENV, API_KEY_ENV, TRACKING_URI_ENV, EXPERIMENT_ENV]
MODEL = "Qwen3.8-27B-oQ4e-mtp"
__version__ = version("tech-market-researcher")

# Enable auto-tracing for LangChain (and LangGraph-based frameworks like Deep Agent)
mlflow.langchain.autolog()

mlflow.set_active_model(name=f"tech-market-researcher-v{__version__}".replace(".", "-"))

console = Console()

def main() -> None:
    missing = [name for name in REQUIRED_ENVS if not os.environ.get(name)]
    if missing:
        print(f"Set {', '.join(missing)} to run the agent.")
        return

    mlflow.set_tracking_uri(os.environ[TRACKING_URI_ENV])
    mlflow.set_experiment(os.environ[EXPERIMENT_ENV])

    llm = ChatOpenAI(
        model=MODEL,
        base_url=os.environ[BASE_URL_ENV],
        api_key=os.environ[API_KEY_ENV],
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
