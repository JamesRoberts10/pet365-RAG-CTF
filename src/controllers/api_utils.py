import os
from dotenv import set_key
from pathlib import Path


env_path = Path(__file__).parent.parent / ".env"


"""
This small module contains the code for managing the LLM API keys.

Key Concepts:

1. Environmental Variables:
   Environmental variables are key-value pairs used for configuration settings, API keys, and other sensitive data that should not be hardcoded into the source code. 
   Storing secrets in this way provides flexibility and prevents us from the embarassment of committing sensitive information to Github.
   Note that environmental variables are only available for the duration of the application's execution. They exist in memory only and are limited to the current run of the application.
   
2. .env Files:
   .env files provide a way to store environment variables persistently. 
   These files are typically placed in the project's root directory and are loaded into the application at runtime. 
   Always remember to include the .env file in your .gitignore file to prevent sensitive information from being committed to Github.


   
3. .gitignore:
   .gitignore is a configuration file used to specify which files or directories should be ignored by Git.
   This might include sensitive information such as API keys and .env files or unnecessary files such as Conda virtual environment files.

"""


# Checks the status of API keys in the environment
# Assigned to the Get API Keys button in the Gradio front end
# Returns the status message for each API key
def get_api_key_status():
    status = []
    for key_name in ["ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GOOGLE_GEMINI_API_KEY"]:
        key = os.getenv(key_name)
        if key:
            status.append(f"{key_name} is set: {key[:7]}...")
        else:
            status.append(f"{key_name} is not set, enter API key above")
    return "\n".join(status)


# Sets API keys for LLMs in the .env file
# Assigned to the Set API Keys button in the Gradio front end
# Builds a status message for each key state in the loop. outputs the full status with new lines.
def set_api_keys(anthropic_key, openai_key, gemini_key):
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
