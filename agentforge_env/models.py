from __future__ import annotations

from typing import Any

from pydantic import Field

try:
    from openenv.core.env_server import Action, Observation, State
except ImportError:  # pragma: no cover
    from openenv_core.env_server import Action, Observation, State


class OversightAction(Action):
    attack_detected: bool
    failure_detected: bool
    goal_degradation: float
    risk_level: str
    violation_types: list[str]
    culprit_span_ids: list[str]
    root_cause: str
    recommended_action: str


class OversightObservation(Observation):
    episode_id: str
    track: str
    difficulty: int
    observation_level: int = 0
    oversight_input: str
    allowed_violation_types: list[str]
    allowed_risk_levels: list[str]
    schema_hint: str
    done_hint: str
    reward: float = 0.0
    done: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)


class OversightState(State):
    episode: dict[str, Any] | None = None
    last_action: dict[str, Any] | None = None
    reward_details: dict[str, Any] = Field(default_factory=dict)
    done: bool = False
    run_id: str = ""
    seed: int | None = None
    filters: dict[str, Any] = Field(default_factory=dict)
    logs: list[dict[str, Any]] = Field(default_factory=list)
    errors: list[dict[str, Any]] = Field(default_factory=list)
