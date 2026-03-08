from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI
try:
    from openenv.core.env_server import create_fastapi_app
except ImportError:  # pragma: no cover
    from openenv_core.env_server import create_fastapi_app

from ..models import OversightAction, OversightObservation
from .environment import AgentForgeOversightEnvironment


DATA_PATH = Path(os.getenv("AGENTFORGE_EPISODES_PATH", Path(__file__).resolve().parents[2] / "data" / "seed_episodes.json"))
SCHEMA_PATH = Path(os.getenv("AGENTFORGE_SCHEMA_PATH", Path(__file__).resolve().parents[2] / "data" / "schema.json"))


def env_factory() -> AgentForgeOversightEnvironment:
    return AgentForgeOversightEnvironment(episodes_path=DATA_PATH, schema_path=SCHEMA_PATH)


app: FastAPI = create_fastapi_app(
    env=env_factory,
    action_cls=OversightAction,
    observation_cls=OversightObservation,
    max_concurrent_envs=8,
)
