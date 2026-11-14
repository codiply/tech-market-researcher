"""Gradio app for chatting with the agent."""

from dotenv import load_dotenv

# Load .env from the current directory (real env vars take precedence)
load_dotenv()

# These imports must run after load_dotenv(): the agent talks to the MLflow
# AI Gateway, whose settings come from .env (real env vars take precedence).
# pylint: disable=wrong-import-position
import gradio as gr

from tech_market_researcher.agents.main import create_agent
from tech_market_researcher.core.arguments import AgentCommandLineArguments, get_args
from tech_market_researcher.core.common import AgentRuntimeContext
from tech_market_researcher.core.constants import DEFAULT_SESSION_ID, DEFAULT_TIME_ZONE, DEFAULT_USER_ID

# pylint: enable=wrong-import-position

GRADIO_APP_PORT = 8899

# Mapping from LangChain message types to the roles Gradio chat items use.
_ROLES = {"human": "user", "ai": "assistant"}


def _to_chat_items(history: list[dict[str, str]]) -> list[dict[str, str]]:
    """Convert the LangChain-role history into Gradio chat items."""
    return [{"role": _ROLES[message["role"]], "content": message["content"]} for message in history]


theme = gr.themes.Default(
    text_size=gr.themes.sizes.text_lg,
)


def main() -> None:
    """Build and launch the web UI."""
    args = get_args(AgentCommandLineArguments)
    agent = create_agent(args)

    context = AgentRuntimeContext(user_id=DEFAULT_USER_ID, session_id=DEFAULT_SESSION_ID, time_zone=DEFAULT_TIME_ZONE)

    def respond(message: str, history: list[dict[str, str]]):
        """Run one turn of the conversation."""
        if not message.strip():
            return history, _to_chat_items(history)
        history = history + [{"role": "user", "content": message}]
        result = agent.invoke(history, context=context)
        history = [
            {"role": message.type, "content": str(message.content)}
            for message in result["messages"]
            if message.type in _ROLES
        ]
        return history, _to_chat_items(history)

    with gr.Blocks(title="Tech Market Researcher", theme=theme) as demo:
        gr.Markdown("# Tech Market Researcher")
        chatbot = gr.Chatbot(height=500)
        history = gr.State([])
        textbox = gr.Textbox(show_label=False, placeholder="Ask the researcher...", autofocus=True)

        events = [textbox, history]
        results = [history, chatbot]
        # Gradio generates submit/click at class creation, so pylint can't see them
        # pylint: disable=no-member
        textbox.submit(respond, inputs=events, outputs=results)
        gr.Button("Submit").click(respond, inputs=events, outputs=results)

    demo.queue().launch(server_name="0.0.0.0", server_port=GRADIO_APP_PORT)


if __name__ == "__main__":
    main()
