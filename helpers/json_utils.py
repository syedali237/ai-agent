import json
import logging
import re


# Some models (gpt-oss) try to call imaginary tools such as "repo_browser" when they want more context.
NO_TOOLS_SYSTEM = (
    "You have no tools, no file access and no internet. Never call a tool or function. "
    "Answer using only the text in the request, and reply with the requested JSON object only."
)


def parse_json(text: str) -> dict:
    """Parse model output as JSON, tolerating ```json fences and stray prose."""
    text = text.strip()
    fence = re.match(r"^```(?:json)?\s*(.*?)\s*```$", text, re.DOTALL)
    if fence:
        text = fence.group(1)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end > start:
            return json.loads(text[start:end + 1])
        raise


def call_validated(client, *, model: str, max_tokens: int, prompt: str, validate, errors, attempts: int = 2):
    """call_json, then `validate(data)`. If validation raises one of `errors`, show the model its own
    error and ask again (models vary run to run; one unlucky reply should not end the task)."""
    last = None
    for _ in range(attempts):
        text = prompt if last is None else (
            f"{prompt}\n\nYour previous answer was rejected: {last}\n"
            "Fix exactly that problem and return the complete JSON again."
        )
        data = call_json(client, model=model, max_tokens=max_tokens, prompt=text)
        try:
            return validate(data)
        except errors as e:
            last = e
            logging.getLogger(__name__).warning("%s reply rejected by validation: %s", model, e)
    raise last


def call_json(client, *, model: str, max_tokens: int, prompt: str) -> dict:
    """Call the model for a JSON object. Retry once on bad JSON, then raise."""
    messages = [{"role": "user", "content": prompt}]
    last_error = None
    for attempt in range(2):
        response = client.messages.create(model=model, max_tokens=max_tokens, messages=messages, system=NO_TOOLS_SYSTEM)
        text = "".join(b.text for b in response.content if getattr(b, "type", "text") == "text")
        try:
            data = parse_json(text)
        except json.JSONDecodeError as e:
            last_error = e
            messages = [{"role": "user", "content": prompt + "\n\nReturn only valid JSON, no prose."}]
            continue
        if not isinstance(data, dict):
            last_error = ValueError("expected a JSON object")
            continue
        return data
    raise ValueError(f"model did not return valid JSON after 2 attempts: {last_error}")
