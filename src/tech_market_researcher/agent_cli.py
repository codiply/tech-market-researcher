"""Interactive chat agent talking to an MLflow AI Gateway endpoint."""

import asyncio

from dotenv import load_dotenv

# Load .env from the current directory (real env vars take precedence)
load_dotenv()

# These imports must run after load_dotenv(): the agent talks to the MLflow
# AI Gateway, whose settings come from .env (real env vars take precedence).
# pylint: disable=wrong-import-position
from rich.console import Console

from tech_market_researcher.agents.main import create_agent
from tech_market_researcher.core.arguments import CliCommandLineArguments, get_args
from tech_market_researcher.core.common import AgentRuntimeContext

# pylint: enable=wrong-import-position

console = Console()


def main() -> None:
    """Run the interactive chat REPL against the configured gateway endpoint."""
    args = get_args(CliCommandLineArguments)
    # One event loop for the whole REPL: the agent's async model client caches its
    # connection pool on the loop it first runs on, so a fresh loop per turn breaks it.
    asyncio.run(_chat(args))


async def _chat(args: CliCommandLineArguments) -> None:
    """Run the interactive chat REPL on a single event loop."""
    agent = create_agent(args)
    context = AgentRuntimeContext(user_id=args.user_id, session_id=args.session_id, time_zone=args.time_zone)
    console.print(f"[dim]Runtime context:\n{context.model_dump_json(indent=2)}[/]")

    # Replay the stored conversation for this session, if any.
    for message in await agent.get_messages(context):
        if message.type == "human":
            console.print(f"[bold red]you> [/bold red]{message.content}")
        elif message.type == "ai":
            console.print(f"[bold cyan]agent>[/bold cyan] {message.content}")
    console.print("Chat with the agent. Type 'exit' to quit.")
    while True:
        user_input = console.input("[bold red]you> [/bold red]").strip()
        if user_input.lower() in {"exit", "quit"}:
            break
        if not user_input:
            continue

        # The checkpointer resumes the conversation, so only the new turn is passed.
        result = await agent.ainvoke([{"role": "user", "content": user_input}], context=context)
        console.print("[bold cyan]agent>[/bold cyan]", result["messages"][-1].content)


if __name__ == "__main__":
    main()
