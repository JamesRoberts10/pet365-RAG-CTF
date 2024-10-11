import os
from dotenv import set_key, load_dotenv
from pathlib import Path


env_path = Path(__file__).parent.parent.parent / ".env"
load_dotenv(env_path)

"""
This module contains the code for managing the Pinecone database.
"""


# Sets the Pinecone API key in the .env file
def set_pinecone_api_key(api_key, index_name):
    set_key(env_path, "PINECONE_API_KEY", api_key)
    set_key(env_path, "PINECONE_INDEX", index_name)
    return "Pinecone API key has been set."


# Sets the Pinecone index name in the .env file
def set_pinecone_index(index_name):
    set_key(env_path, "PINECONE_INDEX", index_name)
    return f"Pinecone index set to: {index_name}"


# Displays the current Pinecone API key and index name
def show_current_config():
    # Reload the environment variables
    load_dotenv(env_path, override=True)

    # Get the latest values
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
    PINECONE_INDEX = os.getenv("PINECONE_INDEX")

    api_key_status = (
        f"Current Pinecone API Key: {PINECONE_API_KEY}"
        if PINECONE_API_KEY
        else "Pinecone API key not set. Enter it above"
    )

    index_status = (
        f"Current Pinecone Index: {PINECONE_INDEX}"
        if PINECONE_INDEX
        else "Pinecone Index not set. Enter it above"
    )

    return f"{api_key_status}\n{index_status}"
