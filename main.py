import os
import sys
from dotenv import load_dotenv
from anthropic import Anthropic
from tools.get_files_info import get_files_info, schema_get_files_info
from tools.write_file import write_file, schema_write_file
from tools.run_python_file import run_python_file, schema_run_python_file
from tools.get_file_content import get_file_content, schema_get_file_content


def main():
    load_dotenv()

    api_key = os.getenv("ANTHROPIC_API_KEY")

    # print(f"Anthropic API Key: {api_key}")
    client = Anthropic(api_key=api_key)

    system_prompt = """You are a helpful AI coding agent.
When a user aks a question or makes a request, make a function call plan. You can perform the following functions:

- List files and directories in a specified directory and provide their sizes and whether they are directories or not.
- Read the content of a specified file.
- Write content to a specified file (Create or update).
- Execute a specified Python file with optional arguments and return the output of the script execution.

All paths provided to you are relative to the working directory. You should not access any files outside of the working directory. If a user provides a path that is outside of the working directory, you should return an error message indicating that the path is not within the working directory.
"""
    

    cli_args = [arg for arg in sys.argv[1:] if arg != "--verbose"]
    verbose_flag = "--verbose" in sys.argv[1:]
    initial_prompt = cli_args[0] if cli_args else None

    messages = []
    session_input_tokens = 0
    session_output_tokens = 0

    print("Interactive mode started. Type 'exit' or 'quit' to stop.")

    pending_prompt = initial_prompt
    while True:
        if pending_prompt is None:
            try:
                pending_prompt = input("You: ").strip()
            except EOFError:
                break
            except KeyboardInterrupt:
                print("\nExiting.")
                break

        if pending_prompt.lower() in {"exit", "quit"}:
            break

        if pending_prompt == "":
            pending_prompt = None
            continue

        messages.append({"role": "user", "content": pending_prompt})
        turn_input_tokens = 0
        turn_output_tokens = 0

        while True:
            message = client.messages.create(
                max_tokens=1024,
                tools=[schema_get_files_info, schema_write_file, schema_run_python_file, schema_get_file_content],
                system=system_prompt,
                messages=messages,
                model="claude-haiku-4-5",
            )

            turn_input_tokens += message.usage.input_tokens
            turn_output_tokens += message.usage.output_tokens
            session_input_tokens += message.usage.input_tokens
            session_output_tokens += message.usage.output_tokens

            assistant_blocks = []
            assistant_text_parts = []
            tool_results = []

            for block in message.content:
                if block.type == "text":
                    assistant_blocks.append({"type": "text", "text": block.text})
                    assistant_text_parts.append(block.text)
                    continue

                if block.type == "tool_use":
                    if verbose_flag:
                        print(f"Calling tool: {block.name} with input: {block.input}")
                    else:
                        print(f"Calling tool: {block.name}")

                    assistant_blocks.append(
                        {
                            "type": "tool_use",
                            "id": block.id,
                            "name": block.name,
                            "input": block.input,
                        }
                    )

                    tool_output = f"Error: Unknown tool '{block.name}'."
                    if block.name == "get_files_info":
                        working_dir = block.input.get("working_dir", os.getcwd())
                        directory = block.input.get("directory", ".")
                        tool_output = get_files_info(working_dir, directory)
                    elif block.name == "write_file":
                        working_dir = block.input.get("working_dir", os.getcwd())
                        file_path = block.input.get("file_path")
                        content = block.input.get("content")
                        if file_path is None or content is None:
                            tool_output = "Error: Missing required inputs for write_file: file_path and content."
                        else:
                            tool_output = write_file(working_dir, file_path, content)
                    elif block.name == "run_python_file":
                        working_dir = block.input.get("working_dir", os.getcwd())
                        file_path = block.input.get("file_path")
                        args = block.input.get("args", [])
                        if file_path is None:
                            tool_output = "Error: Missing required input for run_python_file: file_path."
                        else:
                            tool_output = run_python_file(working_dir, file_path, args)
                    elif block.name == "get_file_content":
                        working_dir = block.input.get("working_dir", os.getcwd())
                        file_path = block.input.get("file_path")
                        if file_path is None:
                            tool_output = "Error: Missing required input for get_file_content: file_path."
                        else:
                            tool_output = get_file_content(working_dir, file_path)

                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": tool_output,
                        }
                    )

            messages.append({"role": "assistant", "content": assistant_blocks})

            if not tool_results:
                print("Assistant:", "\n".join(assistant_text_parts).strip())
                break

            messages.append({"role": "user", "content": tool_results})

        if verbose_flag:
            print(f"User Prompt: {pending_prompt}")
            print(f"Prompt tokens used: {turn_input_tokens}")
            print(f"Response tokens used: {turn_output_tokens}")

        pending_prompt = None

    if verbose_flag:
        print("Session token usage:")
        print(f"Prompt tokens used: {session_input_tokens}")
        print(f"Response tokens used: {session_output_tokens}")


if __name__ == "__main__":
    main()