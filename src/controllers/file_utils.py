import os
from pathlib import Path


# File functions


# Check file exists
def file_exists(filename: str) -> bool:
    return os.path.exists(filename)


# Return file extension of file path (string)
def get_file_extension(filename: str) -> str:
    return Path(filename).suffix


# Read content from local file; probably best to only use text files for now.
def read_content_from_file(filename: str) -> str:
    if file_exists(filename):
        with open(filename, "r") as f:
            content = f.read()
    return content
