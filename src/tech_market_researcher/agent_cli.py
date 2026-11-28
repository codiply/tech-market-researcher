"""Interactive chat agent talking to an MLflow AI Gateway endpoint."""

import asyncio
import logging

from dotenv import load_dotenv

# Load .env from the current directory (real env vars take precedence)
load_dotenv()

# These imports must run after load_dotenv(): the agent talks to the MLflow
# AI Gateway, whose settings come from .env (real env vars take precedence).
# pylint: disable=wrong-import-position
from rich.console import Console

from tech_market_researcher.agents.main import TechMarketResearcherAgent, create_agent
from tech_market_researcher.core.arguments import CliCommandLineArguments, get_args
from tech_market_researcher.core.common import AgentRuntimeContext

# pylint: enable=wrong-import-position

console = Console()
logger = logging.getLogger(__name__)


def main() -> None:
    """Run the interactive chat REPL against the configured gateway endpoint."""
    args = get_args(CliCommandLineArguments)

    async def chat() -> None:
        # Exiting the context releases the agent's async resources (e.g. deletes the sandbox pod),
        # even on Ctrl-C or errors.
        async with await create_agent(args) as agent:
            await _repl(agent, args)

    # One event loop for the whole REPL: the agent's async model client caches its
    # connection pool on the loop it first runs on, so a fresh loop per turn breaks it.
    asyncio.run(chat())


async def _repl(agent: TechMarketResearcherAgent, args: CliCommandLineArguments) -> None:
    """Run the chat REPL loop until the user quits."""
    context = AgentRuntimeContext(user_id=args.user_id, session_id=args.session_id, time_zone=args.time_zone)
    logger.info(
        "Starting chat REPL: user_id=%s, session_id=%s, time_zone=%s",
        args.user_id,
        args.session_id,
        args.time_zone,
    )
    console.print(f"[dim]Runtime context:\n{context.model_dump_json(indent=2)}[/]")

    console.print("Chat with the agent. Type 'exit' to quit.")

    # Replay the stored conversation for this session, if any.
    for message in await agent.get_messages(context):
        if message.type == "human":
            console.print(f"[bold red]you> [/bold red]{message.content}")
        elif message.type == "ai":
            console.print(f"[bold cyan]agent>[/bold cyan] {message.content}")
    while True:
        user_input = console.input("[bold red]you> [/bold red]").strip()
        if user_input.lower() in {"exit", "quit"}:
            break
        if not user_input:
            continue

        # The checkpointer resumes the conversation, so only the new turn is passed.
        result = await agent.ainvoke([{"role": "user", "content": user_input}], context=context)
        logger.info("Chat turn completed: session_id=%s", context.session_id)
        console.print("[bold cyan]agent>[/bold cyan]", result["messages"][-1].content)


if __name__ == "__main__":
    main()
