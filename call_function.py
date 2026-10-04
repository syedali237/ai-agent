import os

from anthropic.types import MessageParam
from tools.get_files_info import get_files_info
from tools.write_file import write_file
from tools.run_python_file import run_python_file
from tools.get_file_content import get_file_content

working_dir = os.getcwd()

def call_function(function_call_part, verbose=False):
    function_name = getattr(function_call_part, "name", None)
    function_input = getattr(function_call_part, "input", {}) or {}

    if verbose:
        print(f"Calling function: {function_name} ({function_input})")
    else:
        print(f"Calling function: {function_name}")

    try:
        if function_name == "get_files_info":
            result = get_files_info(working_dir, function_input.get("directory", "."))
        elif function_name == "write_file":
            file_path = function_input.get("file_path")
            content = function_input.get("content")
            if file_path is None or content is None:
                result = "Error: Missing required inputs for write_file: file_path and content."
            else:
                result = write_file(working_dir, file_path, content)
        elif function_name == "run_python_file":
            file_path = function_input.get("file_path")
            if file_path is None:
                result = "Error: Missing required input for run_python_file: file_path."
            else:
                result = run_python_file(working_dir, file_path, function_input.get("args", []))
        elif function_name == "get_file_content":
            file_path = function_input.get("file_path")
            if file_path is None:
                result = "Error: Missing required input for get_file_content: file_path."
            else:
                result = get_file_content(working_dir, file_path)
        else:
            result = f"Error: Unknown function '{function_name}'."
    except Exception as e:
        result = f"Error: Exception while executing '{function_name}': {str(e)}"

    if not isinstance(result, str):
        result = str(result)

    if verbose and result.startswith("Error:"):
        print(f"Function error: {result}")

    return MessageParam(
        role="tool",
        content=result,
    )