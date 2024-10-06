import gradio as gr
from controllers.ai_utils import query, reload_env_variables
from controllers.api_utils import get_api_key_status, set_api_keys
from controllers.database_utils import (
    set_pinecone_api_key,
    show_current_config,
    set_pinecone_index,
)
from dotenv import load_dotenv
import os
from pathlib import Path

# Add this at the top of the file, after the imports
env_path = Path(__file__).parent.parent.parent / ".env"

# Add this global variable
gpt3_5_message_shown = False


# Add this new function to check if the Pinecone API key is set
def check_api_keys():
    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
    openai_api_key = os.getenv("OPENAI_API_KEY")
    google_api_key = os.getenv("GOOGLE_API_KEY")

    pinecone_key_set = pinecone_api_key is not None and pinecone_api_key.strip() != ""
    llm_key_set = any(
        [
            anthropic_api_key is not None and anthropic_api_key.strip() != "",
            openai_api_key is not None and openai_api_key.strip() != "",
            google_api_key is not None and google_api_key.strip() != "",
        ]
    )

    return pinecone_key_set and llm_key_set


def create_api_key_interface(task_type):
    with gr.Row():
        anthropic_key_input = gr.Textbox(
            label="Anthropic API Key",
            placeholder="Enter your Anthropic API key",
            type="password",
        )
        openai_key_input = gr.Textbox(
            label="OpenAI API Key",
            placeholder="Enter your OpenAI API key",
            type="password",
        )
        gemini_key_input = gr.Textbox(
            label="Google Gemini API Key",
            placeholder="Enter your Google Gemini API key",
            type="password",
        )
    with gr.Row():
        set_keys_btn = gr.Button("Set All API Keys")
    with gr.Row():
        api_key_status = gr.Textbox(label="API Key Status", interactive=False, lines=3)

    set_keys_btn.click(
        fn=set_api_keys,
        inputs=[anthropic_key_input, openai_key_input, gemini_key_input],
        outputs=api_key_status,
    )

    refresh_btn = gr.Button("Refresh API Key Status")
    refresh_btn.click(
        fn=get_api_key_status,
        inputs=[],
        outputs=api_key_status,
    )


def create_initial_interface(task_type):
    global gpt3_5_message_shown

    with gr.Row():
        llm_dropdown = gr.Dropdown(
            choices=["GPT4o", "GPT3_5", "Claude", "Gemini"],
            label="Select Your LLM",
            value="Claude",
        )

    with gr.Row():
        initial_input = gr.Textbox(
            label="Input",
            placeholder="Enter your question here",
        )

    with gr.Row():
        initial_submit_btn = gr.Button("Submit")

    chatbot = gr.Chatbot(label="Chat History", visible=False)
    chat_input = gr.Textbox(
        placeholder="Reply to Pauwde...",
        visible=False,
        label=None,
    )
    chat_submit_btn = gr.Button("Send", visible=False)

    def show_chat_interface(message):
        return {
            initial_input: gr.update(visible=False),
            initial_submit_btn: gr.update(visible=False),
            chatbot: gr.update(visible=True, value=[]),
            chat_input: gr.update(visible=True, label="Input"),
            chat_submit_btn: gr.update(visible=True),
        }

    def chat_response(message, history, llm):
        global gpt3_5_message_shown
        history = history or []
        history.append((message, ""))

        if llm == "GPT3_5" and not gpt3_5_message_shown:
            prepend_message = "Nice work! Using an older model is a good approach as the input sanitisation is often weaker. That is why you were using GPT3.5 right? FLAG{Gipity_Downgrade}\n\n"
            gpt3_5_message_shown = True
        else:
            prepend_message = ""

        for chunk in query(message, llm):
            if history[-1][1] == "":

                history[-1] = (message, prepend_message + chunk)
            else:
                history[-1] = (message, history[-1][1] + chunk)
            yield history, gr.update(
                value="",
                placeholder="Reply to Pauwde...",
            )
        return

    initial_submit_btn.click(
        fn=show_chat_interface,
        inputs=[initial_input],
        outputs=[
            initial_input,
            initial_submit_btn,
            chatbot,
            chat_input,
            chat_submit_btn,
        ],
    ).then(
        fn=chat_response,
        inputs=[initial_input, chatbot, llm_dropdown],
        outputs=[chatbot, chat_input],
    )

    initial_input.submit(
        fn=show_chat_interface,
        inputs=[initial_input],
        outputs=[
            initial_input,
            initial_submit_btn,
            chatbot,
            chat_input,
            chat_submit_btn,
        ],
    ).then(
        fn=chat_response,
        inputs=[initial_input, chatbot, llm_dropdown],
        outputs=[chatbot, chat_input],
    )

    chat_submit_btn.click(
        fn=chat_response,
        inputs=[chat_input, chatbot, llm_dropdown],
        outputs=[chatbot, chat_input],
    )

    chat_input.submit(
        fn=chat_response,
        inputs=[chat_input, chatbot, llm_dropdown],
        outputs=[chatbot, chat_input],
    )

    return (
        initial_input,
        llm_dropdown,
        initial_submit_btn,
        chatbot,
        chat_input,
        chat_submit_btn,
    )


theme = gr.themes.Monochrome(
    spacing_size="md",
    radius_size="sm",
).set(
    embed_radius="*radius_md",
    button_primary_background_fill_hover_dark="*neutral_500",
    button_secondary_background_fill_hover_dark="*neutral_500",
    button_cancel_background_fill_hover_dark="*neutral_500",
)


def create_database_config_interface():
    with gr.Row():
        pinecone_key_input = gr.Textbox(
            label="Pinecone API Key",
            placeholder="Enter your Pinecone API key",
            type="password",
        )
    with gr.Row():
        set_pinecone_key_btn = gr.Button("Set Pinecone API Key")
    with gr.Row():
        index_dropdown = gr.Dropdown(
            label="Pinecone Index",
            choices=["pet365", "super-secret-database"],
            value="pet365",
        )
    with gr.Row():
        show_config_btn = gr.Button("Show Current Database Config")
    with gr.Row():
        database_config_status = gr.Textbox(
            label="Database Config Status", interactive=False, lines=3
        )

    set_pinecone_key_btn.click(
        fn=set_pinecone_api_key,
        inputs=[pinecone_key_input],
        outputs=database_config_status,
    )

    show_config_btn.click(
        fn=show_current_config,
        inputs=[],
        outputs=database_config_status,
    )

    index_dropdown.change(
        fn=set_pinecone_index,
        inputs=[index_dropdown],
        outputs=database_config_status,
    )


def init_interface():
    with gr.Blocks(
        theme=theme,
        css="""
            .root-container {
                padding: 20px 10px !important;
            }
            @media (min-width: 768px) {
                .root-container {
                    padding: 20px 2vw !important;
                }
            }
            @media (min-width: 1200px) {
                .root-container {
                    padding: 20px 4vw !important;
                }
            }
            @media (min-width: 1600px) {
                .root-container {
                    padding: 20px 6vw !important;
                }
            }
            .chatbot {
                height: 400px;
                overflow-y: auto;
            }
            .overlay {
                position: fixed;
                top: 0;
                left: 0;
                width: 100%;
                height: 100%;
                background-color: rgba(0, 0, 0, 0.5);
                z-index: 1000;
                display: flex;
                justify-content: center;
                align-items: center;
            }
            .popup {
                background-color: white;
                padding: 4px;
                border-radius: 2px;
                box-shadow: 0 0 10px rgba(0,0,0,0.1);
                max-width: 100%;
                width: fit-content;
                text-align: center;
            }
            .popup button {
                margin-top: 10px;
            }
            """,
    ) as demo:
        with gr.Column(elem_classes="root-container"):
            gr.Markdown("# Pet365")

            # Create a hidden component to store the reload status
            reload_status = gr.Textbox(visible=False)

            with gr.Tabs() as tabs:
                with gr.TabItem("Chat"):
                    create_initial_interface("Chat")

                with gr.TabItem("LLM Config"):
                    create_api_key_interface("Set API Keys")

                with gr.TabItem("Database Config"):
                    create_database_config_interface()

            # Add an event listener for tab changes
            tabs.select(
                fn=reload_env_variables,
                inputs=[],
                outputs=[reload_status],
            )

            # Add an overlay and popup component
            with gr.Group(visible=False) as overlay:
                with gr.Group(elem_classes="overlay"):
                    with gr.Group(elem_classes="popup"):
                        gr.Markdown("## API Keys Not Set")
                        gr.Markdown(
                            "Please set at least one LLM API key and the Pinecone API key in the Config tabs before using the application."
                        )
                        close_popup_btn = gr.Button("Close")

            # Update the check for API keys
            if not check_api_keys():
                overlay.visible = True

            # Close overlay when the close button is clicked
            close_popup_btn.click(
                fn=lambda: gr.update(visible=False),
                outputs=[overlay],
            )

    return demo


# Ensure environment variables are loaded when the module is imported
load_dotenv(env_path, override=True)
