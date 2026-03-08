from __future__ import annotations

try:
    from openenv.core import EnvClient
    from openenv.core.client_types import StepResult
except ImportError:  # pragma: no cover
    from openenv_core import StepResult
    from openenv_core.env_client import EnvClient

from .models import OversightAction, OversightObservation, OversightState


class AgentForgeEnv(EnvClient[OversightAction, OversightObservation, OversightState]):
    def _step_payload(self, action: OversightAction) -> dict:
        return {
            "attack_detected": action.attack_detected,
            "failure_detected": action.failure_detected,
            "goal_degradation": action.goal_degradation,
            "risk_level": action.risk_level,
            "violation_types": action.violation_types,
            "culprit_span_ids": action.culprit_span_ids,
            "root_cause": action.root_cause,
            "recommended_action": action.recommended_action,
        }

    def _parse_result(self, payload: dict) -> StepResult[OversightObservation]:
        obs = OversightObservation(**payload["observation"])
        return StepResult(
            observation=obs,
            reward=payload.get("reward", obs.reward),
            done=payload.get("done", obs.done),
        )

    def _parse_state(self, payload: dict) -> OversightState:
        return OversightState(**payload)
