import json
import re


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


def call_json(client, *, model: str, max_tokens: int, prompt: str) -> dict:
    """Call the model for a JSON object. Retry once on bad JSON, then raise."""
    messages = [{"role": "user", "content": prompt}]
    last_error = None
    for attempt in range(2):
        response = client.messages.create(model=model, max_tokens=max_tokens, messages=messages)
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
