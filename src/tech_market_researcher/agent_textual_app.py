"""Textual web app for chatting with the agent, served in the browser via textual-serve."""

import asyncio
import sys
import threading
from typing import Any, Coroutine

from dotenv import load_dotenv

# Load .env from the current directory (real env vars take precedence)
load_dotenv()

# These imports must run after load_dotenv(): the agent talks to the MLflow
# AI Gateway, whose settings come from .env (real env vars take precedence).
# pylint: disable=wrong-import-position
from langchain_core.messages import BaseMessage
from rich.text import Text
from textual import work
from textual.app import App, ComposeResult
from textual.containers import Horizontal
from textual.widgets import Button, Footer, Header, Input, OptionList
from textual_serve.server import Server

from tech_market_researcher.agents.main import create_agent
from tech_market_researcher.core.arguments import AgentCommandLineArguments, get_args
from tech_market_researcher.core.common import AgentRuntimeContext, BasePydanticModel
from tech_market_researcher.core.constants import (
    DEFAULT_SESSION_ID,
    DEFAULT_TIME_ZONE,
    DEFAULT_USER_ID,
)
from tech_market_researcher.core.environment import TAILSCALE_HOST_NAME

# pylint: enable=wrong-import-position

TEXTUAL_APP_PORT = 8898

# Mapping from LangChain message types to the label shown for each turn.
_ROLES = {"human": "You", "ai": "Agent"}
_STYLES = {"human": "bold red", "ai": "bold cyan"}


class ChatBotContext(BasePydanticModel):
    """The user/session/time-zone selection shown in the chat UI."""

    time_zone: str
    user_id: str
    session_id: str

    def to_agent_runtime_context(self) -> AgentRuntimeContext:
        """Build the agent's runtime context from this UI selection."""
        return AgentRuntimeContext(
            session_id=self.session_id,
            time_zone=self.time_zone,
            user_id=self.user_id,
        )


def _plain_text(content: str | list[str | dict[str, Any]]) -> str:
    """Flatten a list of content blocks into plain text; plain strings pass through."""
    if isinstance(content, str):
        return content
    return "".join(block["text"] for block in content if isinstance(block, dict) and block.get("type") == "text")


def _message_text(message: BaseMessage) -> Text:
    """Render one conversation turn as a labeled, colored line of text."""
    line = Text()
    line.append(f"{_ROLES[message.type]}: ", style=_STYLES[message.type])
    line.append(_plain_text(message.content))
    return line


class ChatApp(App):
    """The chat UI: a session selector on top, the transcript in the middle, the composer at the bottom."""

    TITLE = "Tech Market Researcher"

    CSS = """
    #session-row {
        height: 3;
        align: left middle;
    }

    #session-row Input {
        width: 1fr;
        margin: 0 1 0 0;
    }

    #composer-row {
        height: 3;
        align: left middle;
    }

    #composer-row Input {
        width: 1fr;
        margin: 0 1 0 0;
    }

    #composer-row Button {
        margin: 0;
    }
    """

    def __init__(self) -> None:
        super().__init__()
        # Each browser session spawns this app in its own subprocess, so the
        # agent is built here, per session, from the loaded config.
        args = get_args(AgentCommandLineArguments)
        self._agent = create_agent(args)
        self._chat_context = ChatBotContext(
            user_id=DEFAULT_USER_ID, session_id=DEFAULT_SESSION_ID, time_zone=DEFAULT_TIME_ZONE
        )
        self._busy = False
        # One event loop for the life of the app, owned by a dedicated thread: the agent's
        # async model client caches its connection pool on the loop it first runs on, so
        # every agent call must share this loop.
        self._agent_loop = asyncio.new_event_loop()
        self._agent_thread = threading.Thread(target=self._agent_loop.run_forever, daemon=True)
        self._agent_thread.start()

    def _agent_result(self, coro: Coroutine[Any, Any, Any]) -> Any:
        """Run a coroutine on the agent's event loop, blocking this thread until it returns."""
        return asyncio.run_coroutine_threadsafe(coro, self._agent_loop).result()

    @property
    def runtime_context(self) -> AgentRuntimeContext:
        """The agent runtime context for the currently selected session."""
        return self._chat_context.to_agent_runtime_context()

    def compose(self) -> ComposeResult:
        """Compose the chat UI."""
        yield Header()
        with Horizontal(id="session-row"):
            yield Input(value=DEFAULT_USER_ID, id="user-id")
            yield Input(value=DEFAULT_SESSION_ID, id="session-id")
            yield Input(value=DEFAULT_TIME_ZONE, id="time-zone")
            yield Button("Apply", id="apply")
        yield OptionList(id="messages")
        with Horizontal(id="composer-row"):
            yield Input(placeholder="Ask the researcher...", id="message")
            yield Button("Submit", variant="primary", id="submit")
        yield Footer()

    def on_mount(self) -> None:
        """Focus the composer and restore the conversation persisted for the default session."""
        self.query_one("#message", Input).focus()
        self._refresh_messages(self._agent_result(self._agent.get_messages(self.runtime_context)))

    def _message_input(self) -> Input:
        """The composer input."""
        return self.query_one("#message", Input)

    def _set_busy(self, busy: bool) -> None:
        """Enable or disable the composer and the session selector while a turn is running."""
        self._busy = busy
        self._message_input().disabled = busy
        self.query_one("#submit", Button).disabled = busy
        if not busy:
            self._message_input().focus()

    def _refresh_messages(self, messages: list[BaseMessage]) -> None:
        """Replace the transcript with the given messages and scroll to the latest turn."""
        transcript = self.query_one("#messages", OptionList)
        options = [_message_text(message) for message in messages if message.type in _ROLES]
        transcript.set_options(options)
        if transcript.option_count:
            transcript.highlighted = transcript.option_count - 1
            transcript.scroll_to_highlight()

    def _submit_message(self) -> None:
        """Send the composer text to the agent, if it is non-empty and no turn is running."""
        message = self._message_input().value.strip()
        if not message or self._busy:
            return
        self._message_input().value = ""
        self._set_busy(True)
        self._run_turn(message)

    def _apply_context(self) -> None:
        """Switch to the conversation thread named by the session selector and load its messages."""
        self._chat_context = ChatBotContext(
            user_id=self.query_one("#user-id", Input).value,
            session_id=self.query_one("#session-id", Input).value,
            time_zone=self.query_one("#time-zone", Input).value,
        )
        self._refresh_messages(self._agent_result(self._agent.get_messages(self.runtime_context)))

    def on_input_submitted(self, event: Input.Submitted) -> None:
        """Submit when Enter is pressed in the composer."""
        if event.input.id == "message":
            self._submit_message()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Route button presses to the right handler."""
        if event.button.id == "submit":
            self._submit_message()
        elif event.button.id == "apply":
            self._apply_context()

    @work(thread=True, exclusive=True, group="agent-turn")
    def _run_turn(self, message: str) -> None:
        """Run one turn of the conversation in a background thread so the UI stays responsive."""
        # The checkpointer resumes the conversation, so only the new turn is passed.
        try:
            result = self._agent_result(
                self._agent.ainvoke([{"role": "user", "content": message}], context=self.runtime_context)
            )
        # Surface any failure in the transcript instead of crashing the session.
        except Exception as error:  # pylint: disable=broad-exception-caught
            self.app.call_from_thread(self._notify, Text(f"Error: {error}", style="bold red"))
            return
        self.app.call_from_thread(self._refresh_messages, result["messages"])
        self.app.call_from_thread(self._set_busy, False)

    def _notify(self, message: Text) -> None:
        """Append a transient line to the transcript, used to surface errors."""
        transcript = self.query_one("#messages", OptionList)
        transcript.add_options([message])
        transcript.highlighted = transcript.option_count - 1
        transcript.scroll_to_highlight()


def main() -> None:
    """Serve the Textual chat app as a web page.

    textual-serve spawns this module in a subprocess per browser connection; that
    subprocess runs the app directly, so the server command must not call this function.
    """
    args = get_args(AgentCommandLineArguments)
    command = f"{sys.executable} -m tech_market_researcher.agent_textual_app --config-name {args.config_name}"
    # Bind all interfaces so the app is reachable over the Tailscale mesh; when a Tailscale
    # host name is configured, advertise it as the public URL so browsers on the mesh build
    # their websocket/static URLs against the right host.
    public_url = f"http://{TAILSCALE_HOST_NAME}:{TEXTUAL_APP_PORT}" if TAILSCALE_HOST_NAME else None
    Server(
        command,
        host="0.0.0.0",
        port=TEXTUAL_APP_PORT,
        title="Tech Market Researcher",
        public_url=public_url,
    ).serve()


if __name__ == "__main__":
    # Entry point for the subprocess spawned by textual-serve: run the app itself.
    ChatApp().run()
