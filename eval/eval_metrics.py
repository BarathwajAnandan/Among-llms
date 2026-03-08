from __future__ import annotations

from typing import Any


def aggregate_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {}

    count = len(rows)
    attack_acc = sum(r["attack_correct"] for r in rows) / count
    failure_acc = sum(r["failure_correct"] for r in rows) / count
    mean_reward = sum(r["reward"] for r in rows) / count
    fp_count = sum(1 for r in rows if r["false_positive"])
    culprit_exact = sum(r["culprit_exact"] for r in rows) / count
    invalid_actions = sum(1 for r in rows if r.get("invalid_action"))
    json_found = sum(1 for r in rows if r.get("json_found"))
    json_only = sum(1 for r in rows if r.get("json_only"))
    schema_valid = sum(1 for r in rows if r.get("schema_valid"))
    by_track: dict[str, dict[str, float]] = {}
    for row in rows:
        track = row["track"]
        summary = by_track.setdefault(track, {"count": 0.0, "mean_reward": 0.0})
        summary["count"] += 1
        summary["mean_reward"] += row["reward"]
    for summary in by_track.values():
        summary["mean_reward"] = summary["mean_reward"] / summary["count"]
    return {
        "count": count,
        "attack_accuracy": attack_acc,
        "failure_accuracy": failure_acc,
        "mean_reward": mean_reward,
        "false_positive_rate": fp_count / count,
        "culprit_exact_rate": culprit_exact,
        "invalid_action_rate": invalid_actions / count,
        "json_found_rate": json_found / count,
        "json_only_rate": json_only / count,
        "schema_valid_rate": schema_valid / count,
        "tracks": by_track,
    }
