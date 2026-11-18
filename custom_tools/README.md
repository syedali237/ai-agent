# Custom tools

Humans add tools here; the agent can never create or register one. A harness can only grant
tools that are registered **and** enabled. Review manifests like code (they live in git).

## Add a tool

1. Write the function in `custom_tools/<module>.py`. Convention (same as `tools/`): the first
   argument is `working_dir`, followed by the arguments declared in `input_schema`, and it returns a string.
2. Add `custom_tools/<name>.json`:

```json
{
  "name": "format_python_file",
  "description": "Format one Python file in place.",
  "enabled": true,
  "access": "mutating",
  "entrypoint": "custom_tools.formatting:format_python_file",
  "path_params": ["file_path"],
  "rollback_strategy": "snapshot_paths",
  "input_schema": {
    "type": "object",
    "properties": {"file_path": {"type": "string", "description": "File to format, relative to the workspace."}},
    "required": ["file_path"]
  }
}
```

## Rules (checked when the registry loads; a bad manifest raises, it is never skipped silently)

- `enabled` must be `true`; a manifest without it is ignored.
- `name`: lowercase letters, digits, underscores. It cannot reuse a built-in name.
- `entrypoint` must live under `custom_tools.` (`module:function`).
- `path_params` lists every argument that holds a file path and must exist in `input_schema.properties`.
  Guardrails resolve and scope these paths, so a tool that touches files must declare them.
- `access: "mutating"` requires `path_params`, and a `rollback_strategy`:
  `snapshot_paths` (the executor snapshots the declared paths before the call) or
  `custom_undo` (also set `undo_entrypoint`, same `module:function` form).
- Mutating custom tools are treated as file writers: a harness must give them an explicit
  `allowed_write` scope. Every custom tool needs a `timeout_seconds` in its harness limits.
