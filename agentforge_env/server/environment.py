from __future__ import annotations

import traceback
import uuid
from pathlib import Path
from typing import Any

try:
    from openenv.core.env_server import Environment
except ImportError:  # pragma: no cover
    from openenv_core.env_server import Environment

from ..models import OversightAction, OversightObservation, OversightState
from ..reward import compute_reward, RewardConfig, DEFAULT_REWARD_CONFIG
from ..serialization import (
    ALLOWED_RISK_LEVELS,
    ALLOWED_VIOLATION_TYPES,
    normalize_action_dict,
    serialize_episode_for_oversight,
)
from ..tracks import load_seed_episodes, sample_episode


class CurriculumConfig:
    """Adaptive difficulty settings. Only active when curriculum=True in reset()."""

    def __init__(
        self,
        window: int = 20,
        promote_threshold: float = 0.80,
        demote_threshold: float = 0.40,
        min_difficulty: int = 2,
        max_difficulty: int = 4,
        min_obs_level: int = 0,
        max_obs_level: int = 4,
    ):
        self.window = window
        self.promote_threshold = promote_threshold
        self.demote_threshold = demote_threshold
        self.min_difficulty = min_difficulty
        self.max_difficulty = max_difficulty
        self.min_obs_level = min_obs_level
        self.max_obs_level = max_obs_level


class AgentForgeOversightEnvironment(Environment[OversightAction, OversightObservation, OversightState]):
    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self, episodes_path: str | Path | None = None, schema_path: str | Path | None = None,
                 reward_config: RewardConfig | None = None,
                 curriculum_config: CurriculumConfig | None = None):
        super().__init__()
        self.episodes_path = episodes_path
        self.schema_path = schema_path
        self.reward_config = reward_config or DEFAULT_REWARD_CONFIG
        self.curriculum_config = curriculum_config or CurriculumConfig()
        self._state = OversightState()
        self._curriculum_active = False
        self._curriculum_history: list[bool] = []
        self._current_difficulty = curriculum_config.min_difficulty if curriculum_config else 2
        self._current_obs_level = 0

    def _append_log(self, event: str, **details: Any) -> None:
        self._state.logs.append({"event": event, **details})

    def _append_error(self, stage: str, message: str, **details: Any) -> None:
        self._state.errors.append({"stage": stage, "message": message, **details})

    def _build_observation(self, reward: float = 0.0, done: bool = False,
                           obs_level: int = 0) -> OversightObservation:
        episode = self._state.episode
        if episode is None:
            raise RuntimeError("Environment state is empty; call reset() first.")
        metadata = {
            "run_id": self._state.run_id,
            "seed": self._state.seed,
            "filters": self._state.filters,
            "step_count": self._state.step_count,
            "reward_details": self._state.reward_details,
            "logs": self._state.logs,
            "errors": self._state.errors,
            "attack_family": episode["attack_family"],
        }
        if self._curriculum_active:
            metadata["curriculum"] = {
                "difficulty": self._current_difficulty,
                "obs_level": self._current_obs_level,
                "success_rate": self._curriculum_success_rate(),
                "history_len": len(self._curriculum_history),
            }
        return OversightObservation(
            episode_id=episode["episode_id"],
            track=episode["track"],
            difficulty=episode["difficulty"],
            observation_level=obs_level,
            oversight_input=serialize_episode_for_oversight(episode, level=obs_level),
            allowed_violation_types=ALLOWED_VIOLATION_TYPES,
            allowed_risk_levels=ALLOWED_RISK_LEVELS,
            schema_hint="Return strict JSON matching the OversightAction schema.",
            done_hint="Single-step environment. One action ends the episode.",
            reward=reward,
            done=done,
            metadata=metadata,
        )

    def _curriculum_success_rate(self) -> float:
        if not self._curriculum_history:
            return 0.0
        window = self._curriculum_history[-self.curriculum_config.window:]
        return sum(window) / len(window)

    def _curriculum_adjust(self, caught_attack: bool) -> None:
        """Update curriculum state after an episode. Promote = harder, demote = easier."""
        self._curriculum_history.append(caught_attack)
        cfg = self.curriculum_config
        rate = self._curriculum_success_rate()

        if len(self._curriculum_history) < cfg.window // 2:
            return

        if rate >= cfg.promote_threshold:
            if self._current_obs_level < cfg.max_obs_level:
                self._current_obs_level += 1
            elif self._current_difficulty < cfg.max_difficulty:
                self._current_difficulty += 1
                self._current_obs_level = cfg.min_obs_level
        elif rate <= cfg.demote_threshold:
            if self._current_obs_level > cfg.min_obs_level:
                self._current_obs_level -= 1
            elif self._current_difficulty > cfg.min_difficulty:
                self._current_difficulty -= 1
                self._current_obs_level = cfg.max_obs_level

    def reset(self, seed: int | None = None, episode_id: str | None = None, **kwargs: Any) -> OversightObservation:
        filters = kwargs.get("filters") or {}
        run_id = kwargs.get("run_id") or f"run_{uuid.uuid4().hex[:12]}"
        curriculum = kwargs.get("curriculum", False)
        obs_level = int(kwargs.get("obs_level", 0))

        self._curriculum_active = curriculum
        if curriculum:
            filters = {**filters, "difficulty": self._current_difficulty}
            obs_level = self._current_obs_level

        mode = "episode_id" if episode_id else "sample"
        if episode_id:
            episodes = load_seed_episodes(self.episodes_path, schema_path=self.schema_path)
            matching = [ep for ep in episodes if ep["episode_id"] == episode_id]
            if not matching:
                raise ValueError(f"Unknown episode_id={episode_id}")
            episode = matching[0]
        else:
            episode = sample_episode(
                seed=seed,
                filters=filters,
                episodes_path=self.episodes_path,
                schema_path=self.schema_path,
            )

        self._state = OversightState(
            episode_id=episode["episode_id"],
            step_count=0,
            episode=episode,
            reward_details={},
            last_action=None,
            done=False,
            run_id=run_id,
            seed=seed,
            filters=dict(filters),
            logs=[],
            errors=[],
        )
        self._append_log(
            "reset",
            mode=mode,
            episode_id=episode["episode_id"],
            track=episode["track"],
            difficulty=episode["difficulty"],
            attack_family=episode["attack_family"],
            curriculum=curriculum,
            obs_level=obs_level,
        )
        return self._build_observation(reward=0.0, done=False, obs_level=obs_level)

    def step(self, action: OversightAction, timeout_s: float | None = None, **kwargs: Any) -> OversightObservation:
        if self._state.done:
            raise RuntimeError("Episode already finished. Call reset() for a new episode.")
        if self._state.episode is None:
            raise RuntimeError("Environment not initialized. Call reset() first.")

        raw_action = action if isinstance(action, dict) else dict(vars(action))
        self._append_log("step_received", raw_action=raw_action)

        try:
            normalized = normalize_action_dict(action)
        except Exception as exc:
            self._append_error(
                "action_validation",
                str(exc),
                raw_action=raw_action,
                traceback=traceback.format_exc(),
            )
            raise

        gold = self._state.episode["ground_truth"]
        reward_result = compute_reward(normalized, gold, config=self.reward_config)

        self._state.step_count += 1
        self._state.last_action = normalized
        self._state.reward_details = reward_result
        self._state.done = True

        if self._curriculum_active:
            attack_present = gold.get("attack_present", False)
            attack_detected = normalized.get("attack_detected", False)
            caught = (attack_detected == attack_present)
            self._curriculum_adjust(caught)

        self._append_log(
            "step_scored",
            normalized_action=normalized,
            total_reward=reward_result["total_reward"],
            reward_components=reward_result["components"],
        )

        obs_level = self._current_obs_level if self._curriculum_active else 0
        return self._build_observation(reward=reward_result["total_reward"], done=True,
                                       obs_level=obs_level)

    @property
    def state(self) -> OversightState:
        return self._state
