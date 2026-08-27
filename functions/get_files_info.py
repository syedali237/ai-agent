import os
from anthropic.types import ToolParam
from config import resolve_path_in_workdir


def get_files_info(working_dir, directory="."):
    abs_working_dir, abs_directory = resolve_path_in_workdir(working_dir, directory)
    final_response = ""

    if os.path.commonpath([abs_working_dir, abs_directory]) != abs_working_dir:
        return f'Error: "{directory}" is not within the working directory.'

    contents = os.listdir(abs_directory)
    for content in contents:
        content_path = os.path.join(abs_directory, content)
        is_dir = os.path.isdir(content_path)
        size = os.path.getsize(content_path)
        final_response += f'- {content}: file_size= {size} bytes, is_dir= {is_dir}\n'

    return final_response

schema_get_files_info: ToolParam = {
    "name": "get_files_info",
    "description": "List files in the specified directory and provide their sizes and whether they are directories or not.",
    "input_schema": {
        "type": "object",
        "properties": {
            "working_dir": {
                "type": "string",
                "description": "The working directory"
            },
            "directory": {
                "type": "string",
                "description": "The directory to list files from (relative to the working directory). Defaults to the working directory if not specified."
            }
        },
        "required": ["working_dir"]
    }
}
