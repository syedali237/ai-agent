import os

from anthropic.types import ToolParam

from config import MAX_CHARS, resolve_path_in_workdir


def get_file_content(working_dir, file_path):
    abs_working_dir, abs_file_path = resolve_path_in_workdir(working_dir, file_path)
    if os.path.commonpath([abs_working_dir, abs_file_path]) != abs_working_dir:
        return f'Error: "{file_path}" is not within the working directory.'

    if not os.path.isfile(abs_file_path):
        return f'Error: "{file_path}" is not a valid file.'

    try:
        with open(abs_file_path, "r") as f:
            file_content_string = f.read(MAX_CHARS)
            if len(file_content_string) >= MAX_CHARS:
                file_content_string += "\n\n[...File content truncated due to size limit.]"
        return file_content_string
    except Exception as e:
        return f'Error: Failed to read "{file_path}". {str(e)}'


schema_get_file_content: ToolParam = {
    "name": "get_file_content",
    "description": "Get the content of the specified file within the working directory.",
    "input_schema": {
        "type": "object",
        "properties": {
            "working_dir": {
                "type": "string",
                "description": "The working directory"
            },
            "file_path": {
                "type": "string",
                "description": "The path to the file to read (relative to the working directory)."
            }
        },
        "required": ["working_dir", "file_path"]
    }
}