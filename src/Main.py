import sys
from pathlib import Path
import webbrowser
import threading
import time

project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.views.frontpage import init_interface


def open_browser():
    # Wait for 2 seconds to allow the server to start
    time.sleep(2)
    webbrowser.open("http://localhost:7860/")


# Main execution
# Access the application via http://localhost:7860/
if __name__ == "__main__":
    demo = init_interface()

    # Start a thread to open the browser
    threading.Thread(target=open_browser, daemon=True).start()

    # Launch the Gradio interface
    demo.launch(server_name="0.0.0.0", server_port=7860)
