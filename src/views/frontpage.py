import gradio as gr
from controllers.ai_utils import query


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

    def chat_response(message, history):
        history = history or []  
        history.append((message, ""))  
        for chunk in query(message):
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
        inputs=[initial_input, chatbot],
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
        inputs=[initial_input, chatbot],
        outputs=[chatbot, chat_input],  
    )

    chat_submit_btn.click(
        fn=chat_response,
        inputs=[chat_input, chatbot],
        outputs=[chatbot, chat_input],  
    )

    chat_input.submit(
        fn=chat_response,
        inputs=[chat_input, chatbot],
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

            create_initial_interface("Query")

    return demo
