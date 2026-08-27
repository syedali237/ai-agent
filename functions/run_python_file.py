import os
import subprocess

from anthropic.types import ToolParam
from config import resolve_path_in_workdir


def run_python_file(working_dir, file_path : str, args : list = []):
    abs_working_dir, abs_file_path = resolve_path_in_workdir(working_dir, file_path)
    if os.path.commonpath([abs_working_dir, abs_file_path]) != abs_working_dir:
        return f'Error: "{file_path}" is not within the working directory.'

    if not os.path.isfile(abs_file_path):
        return f'Error: "{file_path}" is not a valid file.'

    if not file_path.endswith(".py"):
        return f'Error: "{file_path}" is not a Python file.'

    try:
        final_args = ["python", file_path]
        final_args.extend(args)
        output = subprocess.run(final_args, cwd=abs_working_dir, timeout=30, check=True, capture_output=True, text=True)
        final_string = f"""
STDOUT: {output.stdout}
STDERR: {output.stderr}
"""

        if output.stdout == "" and output.stderr == "":
            final_string = "No output was produced by the script."

        if output.returncode != 0:
            final_string += f"\nError: The script exited with a non-zero return code: {output.returncode}"
        
        return final_string
        
    except subprocess.CalledProcessError as e:
        return f'Error: Failed to execute "{file_path}". {str(e)}'
    except subprocess.TimeoutExpired:
        return f'Error: Execution of "{file_path}" timed out.'

schema_run_python_file: ToolParam = {
    "name": "run_python_file",
    "description": "Run the specified Python file within the working directory with the python interpreter. Accepts a list of arguments to pass to the script. Returns the output of the script execution.",
    "input_schema": {
        "type": "object",
        "properties": {
            "working_dir": {
                "type": "string",
                "description": "The working directory"
            },
            "file_path": {
                "type": "string",
                "description": "The path to the Python file to run (relative to the working directory)."
            },
            "args": {
                "type": "array",
                "items": {
                    "type": "string"
                },
                "description": "The arguments to pass to the Python script. An optional array of strings to be used for CLI. Defaults to an empty array if not specified."
            }
        },
        "required": ["working_dir", "file_path", "args"]
    }
}