"""Gradio app for chatting with the agent."""

from collections.abc import Callable
from typing import Any

from dotenv import load_dotenv

# Load .env from the current directory (real env vars take precedence)
load_dotenv()

# These imports must run after load_dotenv(): the agent talks to the MLflow
# AI Gateway, whose settings come from .env (real env vars take precedence).
# pylint: disable=wrong-import-position
import gradio as gr
from langchain_core.messages import BaseMessage

from tech_market_researcher.agents.main import create_agent
from tech_market_researcher.core.arguments import AgentCommandLineArguments, get_args
from tech_market_researcher.core.common import AgentRuntimeContext, BasePydanticModel
from tech_market_researcher.core.constants import DEFAULT_SESSION_ID, DEFAULT_TIME_ZONE, DEFAULT_USER_ID

# pylint: enable=wrong-import-position

GRADIO_APP_PORT = 8899

# Mapping from LangChain message types to the roles Gradio chat items use.
_ROLES = {"human": "user", "ai": "assistant"}

# Button sizing; the apply button is small and understated, the submit button prominent.
CUSTOM_CSS = """
.apply-btn { font-size: 0.8em; padding: 4px 10px; }
.submit-btn { font-size: 1.3em; padding: 12px 28px; }
"""


class ChatBotContext(BasePydanticModel):
    """The user/session/time-zone selection shown in the chat bot UI."""

    user_id: str
    session_id: str
    time_zone: str

    def to_agent_runtime_context(self) -> AgentRuntimeContext:
        """Build the agent's runtime context from this UI selection."""
        return AgentRuntimeContext(
            user_id=self.user_id,
            session_id=self.session_id,
            time_zone=self.time_zone,
        )


def _plain_text(content: str | list[str | dict[str, Any]]) -> str:
    """Flatten a list of content blocks into plain text; plain strings pass through."""
    if isinstance(content, str):
        return content
    return "".join(block["text"] for block in content if isinstance(block, dict) and block.get("type") == "text")


def _to_chat_items(messages: list[BaseMessage]) -> list[dict[str, str]]:
    """Convert the agent's messages into Gradio chat items, keeping only user/assistant turns."""
    return [
        {"role": _ROLES[message.type], "content": _plain_text(message.content)}
        for message in messages
        if message.type in _ROLES
    ]


theme = gr.themes.Default(
    text_size=gr.themes.sizes.text_lg,
)


def _build_ui(
    respond: Callable[[str], tuple[list[dict[str, str]], str, dict[str, Any]]],
    update_submit_button: Callable[[str], dict[str, Any]],
    apply_context: Callable[[str, str, str], list[dict[str, str]]],
    load_chatbot: Callable[[], tuple[list[dict[str, str]], str, str, str, dict[str, Any]]],
) -> gr.Blocks:
    """Build the Gradio UI and wire up its events; callbacks close over the agent and context."""
    with gr.Blocks(title="Tech Market Researcher") as demo:
        gr.Markdown("# Tech Market Researcher")

        with gr.Row():
            user_id_box = gr.Textbox(label="User ID", value=DEFAULT_USER_ID)
            session_id_box = gr.Textbox(label="Session ID", value=DEFAULT_SESSION_ID)
            time_zone_box = gr.Textbox(label="Time zone", value=DEFAULT_TIME_ZONE)
            apply_btn = gr.Button("Apply", variant="secondary", elem_classes=["apply-btn"])

        chatbot = gr.Chatbot(height=500)
        textbox = gr.Textbox(show_label=False, placeholder="Ask the researcher...", autofocus=True)
        submit_btn = gr.Button("Submit", variant="primary", interactive=False, elem_classes=["submit-btn"])

        events = [textbox]
        results = [chatbot, textbox, submit_btn]
        # Gradio generates submit/click at class creation, so pylint can't see them
        # pylint: disable=no-member
        textbox.submit(respond, inputs=events, outputs=results)
        submit_btn.click(respond, inputs=events, outputs=results)
        textbox.change(update_submit_button, inputs=[textbox], outputs=[submit_btn])
        apply_btn.click(
            apply_context,
            inputs=[user_id_box, session_id_box, time_zone_box],
            outputs=[chatbot],
        )

        demo.load(fn=load_chatbot, outputs=[chatbot, user_id_box, session_id_box, time_zone_box, submit_btn])

    return demo


def main() -> None:
    """Build and launch the web UI."""
    args = get_args(AgentCommandLineArguments)
    agent = create_agent(args)
    context = ChatBotContext(user_id=DEFAULT_USER_ID, session_id=DEFAULT_SESSION_ID, time_zone=DEFAULT_TIME_ZONE)

    def load_chatbot() -> tuple[list[dict[str, str]], str, str, str, dict[str, Any]]:
        """Restore the conversation persisted in the checkpointer, if any.

        Also returns the server-side context values so every browser shows the
        session that is currently active on the server, and disables the submit
        button (the textbox starts empty).
        """
        return (
            _to_chat_items(agent.get_messages(context.to_agent_runtime_context())),
            context.user_id,
            context.session_id,
            context.time_zone,
            gr.update(interactive=False),
        )

    def apply_context(user_id: str, session_id: str, time_zone: str) -> list[dict[str, str]]:
        """Switch to the given conversation thread and load the messages stored for it."""
        nonlocal context
        context = ChatBotContext(user_id=user_id, session_id=session_id, time_zone=time_zone)
        return _to_chat_items(agent.get_messages(context.to_agent_runtime_context()))

    def update_submit_button(message: str) -> dict[str, Any]:
        """Disable the submit button while the textbox is empty (or whitespace only)."""
        return gr.update(interactive=bool(message.strip()))

    def respond(message: str) -> tuple[list[dict[str, str]], str, dict[str, Any]]:
        """Run one turn of the conversation; the checkpointer supplies the prior messages.

        Empties the textbox afterwards and keeps the submit button disabled for the
        empty state.
        """
        agent_runtime = context.to_agent_runtime_context()
        if not message.strip():
            return _to_chat_items(agent.get_messages(agent_runtime)), "", gr.update(interactive=False)
        result = agent.invoke([{"role": "user", "content": message}], context=agent_runtime)
        return _to_chat_items(result["messages"]), "", gr.update(interactive=False)

    demo = _build_ui(respond, update_submit_button, apply_context, load_chatbot)

    demo.queue().launch(
        server_name="0.0.0.0",
        server_port=GRADIO_APP_PORT,
        theme=theme,
        css=CUSTOM_CSS,
    )


if __name__ == "__main__":
    main()
