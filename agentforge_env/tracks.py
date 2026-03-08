from __future__ import annotations

import json
import pathlib
import random
from typing import Any

try:
    from jsonschema import Draft202012Validator
except ImportError:  # pragma: no cover
    Draft202012Validator = None


DEFAULT_SEED_PATH = pathlib.Path(__file__).resolve().parents[1] / "data" / "seed_episodes.json"
DEFAULT_SCHEMA_PATH = pathlib.Path(__file__).resolve().parents[1] / "data" / "schema.json"

_EPISODE_VALIDATOR: Draft202012Validator | None = None


def _load_episode_validator(schema_path: str | pathlib.Path | None = None) -> Draft202012Validator | None:
    global _EPISODE_VALIDATOR
    if Draft202012Validator is None:
        return None
    if schema_path is None and _EPISODE_VALIDATOR is not None:
        return _EPISODE_VALIDATOR
    file_path = pathlib.Path(schema_path) if schema_path else DEFAULT_SCHEMA_PATH
    with open(file_path, "r", encoding="utf-8") as f:
        schema = json.load(f)
    validator = Draft202012Validator(schema)
    if schema_path is None:
        _EPISODE_VALIDATOR = validator
    return validator


def validate_episode(episode: dict[str, Any], schema_path: str | pathlib.Path | None = None) -> None:
    validator = _load_episode_validator(schema_path=schema_path)
    if validator is not None:
        validator.validate(episode)


def load_seed_episodes(
    path: str | pathlib.Path | None = None,
    *,
    validate: bool = True,
    schema_path: str | pathlib.Path | None = None,
) -> list[dict[str, Any]]:
    file_path = pathlib.Path(path) if path else DEFAULT_SEED_PATH
    paths = [file_path]
    extra_path = file_path.with_name(f"{file_path.stem}_extra{file_path.suffix}")
    if extra_path.exists():
        paths.append(extra_path)

    episodes: list[dict[str, Any]] = []
    for source_path in paths:
        with open(source_path, "r", encoding="utf-8") as f:
            loaded = json.load(f)
        if not isinstance(loaded, list):
            raise ValueError("Episodes file must contain a JSON list.")
        episodes.extend(loaded)

    if not isinstance(episodes, list):
        raise ValueError("Episodes file must contain a JSON list.")
    if validate:
        for episode in episodes:
            validate_episode(episode, schema_path=schema_path)
    return episodes


def sample_episode(
    seed: int | None = None,
    filters: dict[str, Any] | None = None,
    episodes_path: str | pathlib.Path | None = None,
    *,
    schema_path: str | pathlib.Path | None = None,
) -> dict[str, Any]:
    episodes = load_seed_episodes(episodes_path, validate=True, schema_path=schema_path)
    if filters:
        for key, value in filters.items():
            episodes = [ep for ep in episodes if ep.get(key) == value]
    if not episodes:
        raise ValueError("No episodes matched the requested filters.")
    rng = random.Random(seed)
    return rng.choice(episodes)
