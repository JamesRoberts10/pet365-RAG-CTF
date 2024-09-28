import sys
from pathlib import Path

# Add the project root directory to Python's module search path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.views.frontpage import init_interface

# Main execution
if __name__ == "__main__":
    demo = init_interface()
    demo.launch(server_name="0.0.0.0", server_port=7860)
