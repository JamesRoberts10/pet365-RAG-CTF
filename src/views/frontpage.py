import gradio as gr
from controllers.ai_utils import query
from controllers.api_utils import get_api_key_status, set_api_keys
from controllers.database_utils import (
    set_pinecone_api_key,
    show_current_config,
    set_pinecone_index,
)


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
    with gr.Row():
        llm_dropdown = gr.Dropdown(
            choices=["GPT", "Claude", "Gemini"], label="Select Your LLM", value="Claude"
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
        history = history or []
        history.append((message, ""))
        for chunk in query(message, llm):
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
            choices=["pet365", "Super Secret Database"],
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
            """,
    ) as demo:
        with gr.Column(elem_classes="root-container"):
            gr.Markdown("# Pet365")

            with gr.Tabs():
                with gr.TabItem("Chat"):
                    create_initial_interface("Chat")

                with gr.TabItem("LLM Config"):
                    create_api_key_interface("Set API Keys")

                with gr.TabItem("Database Config"):
                    create_database_config_interface()

    return demo
