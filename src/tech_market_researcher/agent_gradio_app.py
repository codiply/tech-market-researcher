"""Gradio app for chatting with the agent."""

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
from tech_market_researcher.core.common import AgentRuntimeContext
from tech_market_researcher.core.constants import DEFAULT_SESSION_ID, DEFAULT_TIME_ZONE, DEFAULT_USER_ID

# pylint: enable=wrong-import-position

GRADIO_APP_PORT = 8899

# Mapping from LangChain message types to the roles Gradio chat items use.
_ROLES = {"human": "user", "ai": "assistant"}


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


def main() -> None:
    """Build and launch the web UI."""
    args = get_args(AgentCommandLineArguments)
    agent = create_agent(args)
    context = AgentRuntimeContext(user_id=DEFAULT_USER_ID, session_id=DEFAULT_SESSION_ID, time_zone=DEFAULT_TIME_ZONE)

    def load_chatbot() -> list[dict[str, str]]:
        """Restore the conversation persisted in the checkpointer, if any."""
        return _to_chat_items(agent.get_messages(context))

    def apply_context(user_id: str, session_id: str, time_zone: str) -> list[dict[str, str]]:
        """Switch to the given conversation thread and load the messages stored for it."""
        nonlocal context
        context = AgentRuntimeContext(user_id=user_id, session_id=session_id, time_zone=time_zone)
        return _to_chat_items(agent.get_messages(context))

    def respond(message: str) -> list[dict[str, str]]:
        """Run one turn of the conversation; the checkpointer supplies the prior messages."""
        if not message.strip():
            return _to_chat_items(agent.get_messages(context))
        result = agent.invoke([{"role": "user", "content": message}], context=context)
        return _to_chat_items(result["messages"])

    with gr.Blocks(title="Tech Market Researcher") as demo:
        gr.Markdown("# Tech Market Researcher")

        with gr.Row():
            user_id_box = gr.Textbox(label="User ID", value=DEFAULT_USER_ID)
            session_id_box = gr.Textbox(label="Session ID", value=DEFAULT_SESSION_ID)
            time_zone_box = gr.Textbox(label="Time zone", value=DEFAULT_TIME_ZONE)
            apply_btn = gr.Button("Apply", variant="primary")

        chatbot = gr.Chatbot(height=500)
        textbox = gr.Textbox(show_label=False, placeholder="Ask the researcher...", autofocus=True)

        events = [textbox]
        results = [chatbot]
        # Gradio generates submit/click at class creation, so pylint can't see them
        # pylint: disable=no-member
        textbox.submit(respond, inputs=events, outputs=results)
        gr.Button("Submit").click(respond, inputs=events, outputs=results)
        apply_btn.click(
            apply_context,
            inputs=[user_id_box, session_id_box, time_zone_box],
            outputs=[chatbot],
        )

        demo.load(fn=load_chatbot, outputs=[chatbot])

    demo.queue().launch(server_name="0.0.0.0", server_port=GRADIO_APP_PORT, theme=theme)


if __name__ == "__main__":
    main()
