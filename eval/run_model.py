from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agentforge_env.inference import generate_oversight_action_with_metadata
from agentforge_env.reward import compute_reward, invalid_action_reward
from agentforge_env.server.environment import AgentForgeOversightEnvironment
from agentforge_env.tracks import load_seed_episodes
from eval.eval_metrics import aggregate_metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=str, required=True)
    parser.add_argument("--schema", type=str, default=str(ROOT / "data" / "schema.json"))
    parser.add_argument("--base_url", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--max_episodes", type=int, default=0)
    parser.add_argument("--out", type=str, default="model_metrics.json")
    parser.add_argument("--run_id", type=str, default="remote_model_eval")
    args = parser.parse_args()

    episodes = load_seed_episodes(args.episodes, schema_path=args.schema)
    if args.max_episodes > 0:
        episodes = episodes[: args.max_episodes]
    env = AgentForgeOversightEnvironment(episodes_path=args.episodes, schema_path=args.schema)

    rows: list[dict[str, Any]] = []
    for episode in episodes:
        obs = env.reset(episode_id=episode["episode_id"], run_id=args.run_id)
        invalid_action = False
        raw_error = None
        raw_output = None
        parse_meta = {"json_found": False, "json_only": False, "schema_valid": False}
        try:
            result = generate_oversight_action_with_metadata(
                oversight_input=obs.oversight_input,
                base_url=args.base_url,
                model=args.model,
            )
            action = result["action"]
            raw_output = result["raw_output"]
            parse_meta = result["parse_meta"]
            env.step(action)
            reward_details = compute_reward(action, episode["ground_truth"], parse_meta=parse_meta)
            reward = reward_details["total_reward"]
        except Exception as exc:
            invalid_action = True
            raw_error = str(exc)
            reward_details = invalid_action_reward(raw_error)
            reward = reward_details["total_reward"]
            action = None

        row = {
            "run_id": args.run_id,
            "episode_id": episode["episode_id"],
            "track": episode["track"],
            "difficulty": episode["difficulty"],
            "attack_family": episode["attack_family"],
            "reward": reward,
            "attack_correct": int(
                action is not None and action["attack_detected"] == episode["ground_truth"]["attack_present"]
            ),
            "failure_correct": int(
                action is not None
                and action["failure_detected"] == episode["ground_truth"]["defender_compromised"]
            ),
            "culprit_exact": int(
                action is not None
                and set(action["culprit_span_ids"]) == set(episode["ground_truth"]["culprit_span_ids"])
            ),
            "false_positive": bool(
                action is not None and action["attack_detected"] and not episode["ground_truth"]["attack_present"]
            ),
            "invalid_action": invalid_action,
            "json_found": parse_meta["json_found"],
            "json_only": parse_meta["json_only"],
            "schema_valid": parse_meta["schema_valid"],
            "prediction": action,
            "raw_output": raw_output,
            "error": raw_error,
            "reward_components": reward_details["components"],
        }
        rows.append(row)

    metrics = aggregate_metrics(rows)
    output = {"mode": "remote_model", "run_id": args.run_id, "metrics": metrics, "per_episode": rows}
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
