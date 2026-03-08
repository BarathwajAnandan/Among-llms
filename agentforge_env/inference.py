from __future__ import annotations

import json
import os
import urllib.request
from typing import Any

from .serialization import parse_oversight_response, repair_action_dict

DEFAULT_SYSTEM_PROMPT = (
    "You are an oversight agent. Return only one strict JSON object with exactly the "
    "required oversight fields. Do not add markdown, explanations, or extra text."
)


def call_openai_compatible(
    *,
    base_url: str,
    model: str,
    oversight_input: str,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    api_key: str | None = None,
    max_tokens: int = 256,
    temperature: float = 0.0,
    timeout_s: float = 60.0,
) -> dict[str, Any]:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": oversight_input},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    headers = {"Content-Type": "application/json"}
    token = api_key or os.getenv("OPENAI_API_KEY")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
    )
    with urllib.request.urlopen(request, timeout=timeout_s) as response:
        return json.loads(response.read().decode("utf-8"))


def generate_oversight_action(
    *,
    oversight_input: str,
    base_url: str,
    model: str,
    api_key: str | None = None,
    max_attempts: int = 2,
) -> dict[str, Any]:
    result = generate_oversight_action_with_metadata(
        oversight_input=oversight_input,
        base_url=base_url,
        model=model,
        api_key=api_key,
        max_attempts=max_attempts,
    )
    return result["action"]


def generate_oversight_action_with_metadata(
    *,
    oversight_input: str,
    base_url: str,
    model: str,
    api_key: str | None = None,
    max_attempts: int = 2,
) -> dict[str, Any]:
    last_error: Exception | None = None
    last_text = ""
    last_meta: dict[str, Any] | None = None
    for attempt in range(max_attempts):
        system_prompt = DEFAULT_SYSTEM_PROMPT
        if attempt > 0:
            system_prompt += " Retry because your last answer was invalid. Output JSON only."
        response = call_openai_compatible(
            base_url=base_url,
            model=model,
            oversight_input=oversight_input,
            system_prompt=system_prompt,
            api_key=api_key,
        )
        last_text = response["choices"][0]["message"]["content"]
        action, meta = parse_oversight_response(last_text)
        last_meta = meta
        if action is not None and meta["schema_valid"]:
            repaired = repair_action_dict(action, oversight_input)
            return {"action": repaired, "raw_output": last_text, "parse_meta": meta, "attempts": attempt + 1}
        last_error = ValueError(str(meta["normalization_error"]))
    detail = f" Last response: {last_text}" if last_text else ""
    if last_meta is not None:
        detail += f" Parse meta: {json.dumps(last_meta, ensure_ascii=False)}"
    raise RuntimeError(f"Failed to produce a valid oversight action.{detail}") from last_error
