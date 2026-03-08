"""Convert agentforge precomputed eval outputs to the format expected by
build-overseer-input.ts.

Reads precomputed_episode_outputs.jsonl (labels: 0.5B-base, 0.5B-RL) and
banking_episodes_249.json, outputs two metrics JSON files the animation
pipeline can ingest directly.

Usage:
    python scripts/convert_eval_to_viz.py \
        --precomputed ../outputs/evals/precomputed_episode_outputs.jsonl \
        --episodes ../data/banking_episodes_249.json \
        --out-dir data/generated
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_episode_index(episodes_path: str) -> dict:
    with open(episodes_path) as f:
        episodes = json.load(f)
    return {ep["episode_id"]: ep for ep in episodes}


def build_metrics_file(rows: list[dict], ep_index: dict, name: str, run_id: str) -> dict:
    n = len(rows)
    attack_correct = 0
    failure_correct = 0
    total_reward = 0.0
    valid_count = 0
    per_episode = []

    for row in rows:
        eid = row["episode_id"]
        ep = ep_index.get(eid, {})
        gt = ep.get("ground_truth", {})
        action = row.get("action") or {}
        components = row.get("components", {})
        reward = row.get("reward", 0)
        valid = row.get("valid", False)

        pred_attack = action.get("attack_detected")
        gold_attack = gt.get("attack_present", gt.get("attack_detected", False))
        pred_failure = action.get("failure_detected")
        gold_failure = gt.get("failure_present", gt.get("failure_detected", False))

        a_correct = pred_attack == gold_attack if isinstance(pred_attack, bool) else False
        f_correct = pred_failure == gold_failure if isinstance(pred_failure, bool) else False
        if a_correct:
            attack_correct += 1
        if f_correct:
            failure_correct += 1
        total_reward += reward
        if valid:
            valid_count += 1

        culprit_pred = set(action.get("culprit_span_ids") or [])
        culprit_gold = set(gt.get("culprit_span_ids") or [])
        culprit_exact = culprit_pred == culprit_gold if culprit_gold else len(culprit_pred) == 0

        per_episode.append({
            "episode_id": eid,
            "track": ep.get("track", "unknown"),
            "difficulty": ep.get("difficulty", 0),
            "attack_family": ep.get("attack_family", "unknown"),
            "reward": reward,
            "attack_correct": a_correct,
            "failure_correct": f_correct,
            "culprit_exact": culprit_exact,
            "false_positive": (pred_attack is True and gold_attack is False),
            "invalid_action": not valid,
            "json_found": valid,
            "json_only": valid,
            "schema_valid": valid,
            "prediction": action,
            "reward_components": components,
            "raw_output": row.get("raw_output", ""),
        })

    metrics = {
        "count": n,
        "attack_accuracy": attack_correct / max(n, 1),
        "failure_accuracy": failure_correct / max(n, 1),
        "mean_reward": total_reward / max(n, 1),
        "false_positive_rate": 0,
        "culprit_exact_rate": sum(1 for e in per_episode if e["culprit_exact"]) / max(n, 1),
        "invalid_action_rate": (n - valid_count) / max(n, 1),
        "json_found_rate": valid_count / max(n, 1),
        "json_only_rate": valid_count / max(n, 1),
        "schema_valid_rate": valid_count / max(n, 1),
    }

    return {
        "mode": "remote_model",
        "run_id": run_id,
        "metrics": metrics,
        "per_episode": per_episode,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--precomputed", required=True)
    parser.add_argument("--episodes", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--max-episodes", type=int, default=0, help="Limit episodes per run (0=all)")
    args = parser.parse_args()

    with open(args.precomputed) as f:
        all_rows = [json.loads(l) for l in f if l.strip()]

    ep_index = load_episode_index(args.episodes)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    by_label = {}
    for row in all_rows:
        label = row.get("label", "unknown")
        by_label.setdefault(label, []).append(row)

    if args.max_episodes > 0:
        for label in by_label:
            by_label[label] = by_label[label][:args.max_episodes]

    label_map = {
        "0.5B-base": ("weak_metrics.json", "0.5B base (before RL)", "qwen25-05b-base"),
        "0.5B-RL":   ("trained_metrics.json", "0.5B RL (after GRPO)", "qwen25-05b-rl"),
    }

    for label, rows in by_label.items():
        if label not in label_map:
            print(f"Skipping unknown label: {label}")
            continue

        filename, name, run_id = label_map[label]
        result = build_metrics_file(rows, ep_index, name, run_id)
        out_path = out_dir / filename
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)

        m = result["metrics"]
        print(f"{name}: {len(rows)} episodes, "
              f"attack_acc={m['attack_accuracy']:.1%}, "
              f"mean_reward={m['mean_reward']:.2f} → {out_path}")


if __name__ == "__main__":
    main()
