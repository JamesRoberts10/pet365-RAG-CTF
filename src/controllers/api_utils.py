import os
from dotenv import set_key
from pathlib import Path


env_path = Path(__file__).parent.parent / ".env"


"""
This small module contains the code for managing the LLM API keys.

Key Concepts:

1. Environmental Variables:
  
2. .env Files:
 
3. .gitignore:
   

"""


def get_api_key_status():
    # Checks the status of API keys in the environment
    # Assigned to the Get API Keys button in the Gradio front end
    # Returns the status message for each API key
    status = []
    for key_name in ["ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GOOGLE_GEMINI_API_KEY"]:
        key = os.getenv(key_name)
        if key:
            status.append(f"{key_name} is set: {key[:7]}...")
        else:
            status.append(f"{key_name} is not set, enter API key above")
    return "\n".join(status)


def set_api_keys(anthropic_key, openai_key, gemini_key):
    # Sets API keys for LLMs in the .env file
    # Assigned to the Set API Keys button in the Gradio front end
    # Builds a status message for each key state in the loop. outputs the full status with new lines.

    status = []
    key_pairs = [
        ("ANTHROPIC_API_KEY", anthropic_key),
        ("OPENAI_API_KEY", openai_key),
        ("GOOGLE_GEMINI_API_KEY", gemini_key),
    ]

    for key_name, key in key_pairs:
        if key:
            os.environ[key_name] = key
            set_key(env_path, key_name, key)
            status.append(f"{key_name} set successfully: {key[:7]}...")
        else:
            status.append(f"{key_name} Not currently set...")

    return "\n".join(status)
