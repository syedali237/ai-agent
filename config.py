import os


MAX_CHARS = 10000  # Maximum number of characters to read from the file

# Model IDs live here only. Different models for different jobs: the model that
# executes a step must never be the only one to judge it.
INTAKE_MODEL = "claude-haiku-4-5"      # small/cheap: normalize the request
ANALYSIS_MODEL = "claude-opus-5-5"     # large: decompose into substeps
REVIEW_MODEL = "claude-sonnet-5-5"     # second large: harness gen + plan review
EXECUTION_MODEL = "claude-haiku-4-5"   # small/fast: run one step in its harness
VERIFY_MODEL = "claude-opus-5-5"       # large: compare plan vs. what happened


def resolve_path_in_workdir(working_dir, input_path):
	abs_working_dir = os.path.abspath(os.path.expanduser(working_dir))
	candidate_path = os.path.expanduser(input_path)

	if os.path.isabs(candidate_path):
		abs_path = os.path.abspath(candidate_path)
	else:
		normalized = os.path.normpath(candidate_path)
		working_dir_name = os.path.basename(abs_working_dir)
		prefix = f"{working_dir_name}{os.sep}"
		if normalized == working_dir_name:
			normalized = "."
		elif normalized.startswith(prefix):
			normalized = normalized[len(prefix):]
		abs_path = os.path.abspath(os.path.join(abs_working_dir, normalized))

	return abs_working_dir, abs_path