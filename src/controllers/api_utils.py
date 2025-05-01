import os
from dotenv import set_key
from pathlib import Path


env_path = Path(__file__).parent.parent / ".env"


"""
This module contains utility functions for managing LLM API keys
and environment variables.

Key Concepts:

1. Environment Variables:
   Environment variables are key-value pairs external to the application's source code,
   often used for configuration settings, API keys, and other sensitive data.
   Storing secrets this way provides security and flexibility, avoiding the risk of
   accidentally committing sensitive information to Git.
   Note: Standard environment variables typically exist only for the duration of the
   application's process execution (in memory) unless set system-wide.

2. .env Files:
   `.env` files provide a conventional way to define environment variables for a specific
   project, storing them persistently in a file within the project directory.
   Libraries (like `python-dotenv`) can load these variables into the application's
   environment at runtime.
   Crucially, the `.env` file itself should always be added to the project's
   `.gitignore` file to prevent committing secrets.

3. .gitignore:
   A `.gitignore` file is a text file used by the Git version control system to specify
   intentionally untracked files or directories that Git should ignore.
   Common examples include `.env` files containing secrets, compiled code, log files,
   and virtual environment directories (e.g., `venv/`, `__pycache__/`). This ensures
   only relevant source code and configuration templates are tracked in the repository.
"""


# Checks the current status of required API keys within the environment variables.
# This function is linked to a button in the Gradio frontend
def get_api_key_status():
    """
    Checks if specified API keys are set in the environment variables.

    Returns:
        str: A multi-line string indicating the status (set or not set)
             of each checked API key. Shows the first 7 characters if set.
    """
    # List to hold status messages for each key.
    status = []
    for key_name in ["ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GOOGLE_GEMINI_API_KEY"]:
        key = os.getenv(key_name)
        if key:
            # If the key is set, append a status message showing the key name and the first 7 characters.
            # Masking the key prevents accidental exposure of the full key in logs or UI.
            status.append(f"{key_name} is set: {key[:7]}...")
        else:
            # If the key is not set, append a message indicating this.
            status.append(f"{key_name} is not set, enter API key above")
    # Join the list of status messages into a single string, separated by newlines.
    return "\n".join(status)


# Sets the provided API keys both in the current environment session and persistently in the .env file.
# This function is linked to a button in the Gradio frontend
def set_api_keys(anthropic_key, openai_key, gemini_key):
    """
    Sets the provided API keys in the environment variables for the current session
    and updates the .env file for persistence.

    Args:
        anthropic_key: The API key for Anthropic.
        openai_key: The API key for OpenAI.
        google_key: The API key for Google (Gemini/Vertex AI).

    Returns:
        str: A multi-line string confirming which keys were set or indicating
             if they were left unset. Shows the first 7 characters if set.
    """
    status = []
    key_pairs = [
        ("ANTHROPIC_API_KEY", anthropic_key),
        ("OPENAI_API_KEY", openai_key),
        ("GOOGLE_GEMINI_API_KEY", gemini_key),
    ]

    for key_name, key in key_pairs:
        if key:
            # If a key value is provided:
            # 1. Set the environment variable for the *current running process*.
            #    This makes the key immediately available to the application without a restart.
            os.environ[key_name] = key
            # 2. Update the key in the .env file for persistence across application restarts.
            #    `set_key` from python-dotenv handles adding or updating the key in the file.
            set_key(env_path, key_name, key)
            # Append a success message, showing the first 7 characters of the set key.
            status.append(f"{key_name} set successfully: {key[:7]}...")
        else:
            status.append(f"{key_name} Not currently set...")

    return "\n".join(status)
