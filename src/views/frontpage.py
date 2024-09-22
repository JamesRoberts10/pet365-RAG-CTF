import gradio as gr
from controllers.ai_utils import query


def create_initial_interface(task_type):
    with gr.Row():
        llm_dropdown = gr.Dropdown(
            choices=["GPT", "Claude", "Gemini"], label="Select Your LLM", value="Claude"
        )

    with gr.Row():
        text_input = gr.Textbox(label="Input", placeholder="Enter your question here")

    with gr.Row():
        submit_btn = gr.Button("Submit")

    with gr.Row(visible=False) as output_row:
        output_box = gr.Textbox(
            label="",
            show_label=False,
            interactive=False,
            elem_classes="expanding-output",
        )

    def show_output(*args):
        return {output_row: gr.update(visible=True)}

    def stream_output(text_input):
        full_response = ""
        for chunk in query(text_input):
            full_response += chunk
            yield full_response

    submit_btn.click(fn=show_output, inputs=[], outputs=[output_row], queue=False).then(
        fn=stream_output, inputs=[text_input], outputs=[output_box], api_name="query"
    )
    text_input.submit(
        fn=show_output, inputs=[], outputs=[output_row], queue=False
    ).then(
        fn=stream_output, inputs=[text_input], outputs=[output_box], api_name="query"
    )

    return text_input, llm_dropdown, submit_btn, output_box


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
            .compact-file {
                max-height: 25vh;
                overflow-y: auto;
            }
            .compact-file > .file-preview {
                max-height: 25vh;
            }
            .compact-file > .file-preview img {
                max-height: 25vh;
            }
            .expanding-output {
                margin-top: 20px;
            }
            .expanding-output textarea {
                min-height: 100px;
                max-height: 400px;
                height: auto;
                overflow-y: auto;
                resize: none;
                padding: 10px !important;
                border: 1px solid rgba(0, 0, 0, 0.1) !important;
                border-radius: 4px !important;
                font-size: 14px !important;
            }
            .generating {
                border: none;
            }
            """,
    ) as demo:
        with gr.Column(elem_classes="root-container"):
            gr.Markdown("# Pet365")

            create_initial_interface("Query")

    return demo
