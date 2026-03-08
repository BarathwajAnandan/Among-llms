from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agentforge_env.reward import compute_reward, invalid_action_reward
from agentforge_env.serialization import parse_oversight_response, repair_action_dict


def reward_from_completion(completion: str, gold_episode: dict[str, Any]) -> float:
    action, meta = parse_oversight_response(completion)
    if action is None or not meta["schema_valid"]:
        return float(invalid_action_reward()["total_reward"])
    repaired_action = repair_action_dict(action, gold_episode["scenario"].get("oversight_input", ""), calibration="conservative")
    result = compute_reward(repaired_action, gold_episode["ground_truth"], parse_meta=meta)
    return float(result["total_reward"])
