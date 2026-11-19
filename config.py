import os

from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv(usecwd=True))  # so the env overrides below see .env

MAX_CHARS = 10000  # Maximum number of characters to read from the file

# --- Model providers -----------------------------------------------------------------------
# Models listed in PROVIDERS are called over an OpenAI-compatible chat-completions API
# (helpers/llm.py). Any other model id (e.g. "claude-...") goes to the Anthropic SDK.
# Everything is overridable from .env; see .env.example. Verify the model ids with your provider.
GPT_OSS_MODEL = os.getenv("GPT_OSS_MODEL", "openai/gpt-oss-120b")     # id differs per host
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")          # check DeepSeek's current ids

PROVIDERS = {
    GPT_OSS_MODEL: {
        "base_url": os.getenv("GPT_OSS_BASE_URL", "https://openrouter.ai/api/v1"),
        "api_key_env": "GPT_OSS_API_KEY",
    },
    DEEPSEEK_MODEL: {
        "base_url": os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
        "api_key_env": "DEEPSEEK_API_KEY",
    },
}

# Model IDs per role live here only. The model that executes a step must never be the only one
# to judge it, and the plan reviewer must not be the plan's author (helpers/llm.py enforces both).
INTAKE_MODEL = GPT_OSS_MODEL       # small/cheap: normalize the request
ANALYSIS_MODEL = DEEPSEEK_MODEL    # large: decompose into substeps
HARNESS_MODEL = DEEPSEEK_MODEL     # large: per-step guardrails
REVIEW_MODEL = GPT_OSS_MODEL       # independent of the plan's author: plan review
EXECUTION_MODEL = GPT_OSS_MODEL    # fast: run one step in its harness
VERIFY_MODEL = DEEPSEEK_MODEL      # independent of the executor: judge what happened


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