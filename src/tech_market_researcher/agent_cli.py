"""Interactive chat agent talking to an MLflow AI Gateway endpoint."""

from importlib.metadata import version

from dotenv import load_dotenv

# Load .env from the current directory (real env vars take precedence)
load_dotenv()

# These imports must run after load_dotenv(): the agent talks to the MLflow
# AI Gateway, whose settings come from .env (real env vars take precedence).
# pylint: disable=wrong-import-position
from rich.console import Console

from tech_market_researcher.agents.main import create_agent
from tech_market_researcher.core.arguments import CliCommandLineArguments, get_args
from tech_market_researcher.core.common import AgentContext
from tech_market_researcher.core.observability import setup_observability

# pylint: enable=wrong-import-position

__version__ = version("tech-market-researcher")

console = Console()


def main() -> None:
    """Run the interactive chat REPL against the configured gateway endpoint."""
    args = get_args(CliCommandLineArguments)
    setup_observability(__version__)

    agent = create_agent(args)
    context = AgentContext(user_id=args.user_id, session_id=args.session_id, time_zone=args.time_zone)

    console.print("Chat with the agent. Type 'exit' to quit.")
    messages: list = []
    while True:
        user_input = console.input("[bold red]you> [/bold red]").strip()
        if user_input.lower() in {"exit", "quit"}:
            break
        if not user_input:
            continue

        messages.append({"role": "user", "content": user_input})
        result = agent.invoke({"messages": messages}, context=context)
        # Keep the full conversation (incl. the agent's reply) for the next turn
        messages = result["messages"]
        console.print("[bold cyan]agent>[/bold cyan]", result["messages"][-1].content)


if __name__ == "__main__":
    main()
