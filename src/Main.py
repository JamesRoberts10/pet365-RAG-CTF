from views.frontpage import init_interface

# Main execution
if __name__ == "__main__":
    demo = init_interface()
    demo.launch(server_name="0.0.0.0", server_port=7860)
