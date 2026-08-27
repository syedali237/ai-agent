# Coding Agent

An interactive AI-powered coding assistant that leverages Claude to help with file operations, code execution, and project management through natural language commands.

This project provides a CLI chat loop where you can ask for tasks like listing files, reading file contents, writing files, and running Python scripts inside the current working directory. The agent can call structured tools, execute them safely, and return human-readable responses in the terminal.

## Functions It Can Perform

- List files and directories in a target folder, including file size and whether each item is a directory.
- Read file content from a path inside the working directory.
- Write or update file content inside the working directory.
- Run Python files with optional CLI arguments and return stdout/stderr.

## Example Usage

![Coding Agent CLI Example](example.png)
