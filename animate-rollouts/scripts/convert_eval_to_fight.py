"""Convert precomputed eval data into RolloutWrestling fight events.

Each episode becomes a fight step: the attacker's injection vs the oversight
model's verdict. compromised = attack missed, taskSuccess = attack caught.

Usage:
    python scripts/convert_eval_to_fight.py \
        --precomputed ../outputs/evals/precomputed_episode_outputs.jsonl \
        --episodes ../data/banking_episodes_249.json \
        --out data/generated/fight-input.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ATTACKER_ACTIONS = [
    "inject_payload", "spawn_phishing_trap", "stealth_probe",
    "payload_drop", "inject_noise", "exfiltrate_data",
    "forge_authority", "corrupt_format", "sabotage_goal",
    "priority_hijack", "social_engineer", "credential_steal",
]

DEFENDER_ACTIONS = [
    "containment_lock", "adaptive_policy", "integrity_audit",
    "deep_scan", "anomaly_flag", "rollback_action",
    "quarantine_span", "trust_verify", "pattern_match",
    "context_check", "sandbox_isolate", "evidence_collect",
]


def attack_family_to_action(family: str, idx: int) -> str:
    family_map = {
        "direct_instruction_injection": "inject_payload",
        "phishing_prompt_injection": "spawn_phishing_trap",
        "goal_hijacking": "sabotage_goal",
        "data_exfiltration": "exfiltrate_data",
        "authority_spoofing": "forge_authority",
        "context_manipulation": "inject_noise",
        "priority_manipulation": "priority_hijack",
    }
    for key, action in family_map.items():
        if key in family.lower():
            return action
    return ATTACKER_ACTIONS[idx % len(ATTACKER_ACTIONS)]


def verdict_to_defender_action(caught: bool, idx: int) -> str:
    if caught:
        strong = ["containment_lock", "anomaly_flag", "quarantine_span",
                   "deep_scan", "integrity_audit", "evidence_collect"]
        return strong[idx % len(strong)]
    weak = ["adaptive_policy", "context_check", "pattern_match",
            "trust_verify", "sandbox_isolate", "rollback_action"]
    return weak[idx % len(weak)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--precomputed", required=True)
    parser.add_argument("--episodes", required=True)
    parser.add_argument("--label", default="0.5B-RL",
                        help="Which label to use from precomputed (0.5B-base or 0.5B-RL)")
    parser.add_argument("--title", default="Attacker vs Oversight Agent")
    parser.add_argument("--attacker-name", default="Prompt Injector")
    parser.add_argument("--defender-name", default="Oversight Agent (RL)")
    parser.add_argument("--character-style", default="among-us", choices=["among-us", "sprite"])
    parser.add_argument("--max-episodes", type=int, default=0)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    with open(args.precomputed) as f:
        all_rows = [json.loads(l) for l in f if l.strip()]

    with open(args.episodes) as f:
        ep_list = json.load(f)
    ep_index = {ep["episode_id"]: ep for ep in ep_list}

    rows = [r for r in all_rows if r.get("label") == args.label]
    if args.max_episodes > 0:
        rows = rows[:args.max_episodes]

    events = []
    episode_num = 0
    current_track = None

    for idx, row in enumerate(rows):
        eid = row["episode_id"]
        ep = ep_index.get(eid, {})
        gt = ep.get("ground_truth", {})
        action = row.get("action") or {}
        track = ep.get("track", "banking")

        if track != current_track:
            episode_num += 1
            current_track = track

        attack_present = gt.get("attack_present", gt.get("attack_detected", True))
        pred_attack = action.get("attack_detected", False)
        caught = (pred_attack == True) if attack_present else True

        env_name = f"{track.title()}-D{ep.get('difficulty', 2)}"
        attacker_action = attack_family_to_action(ep.get("attack_family", ""), idx)
        defender_action = verdict_to_defender_action(caught, idx)

        events.append({
            "episode": episode_num,
            "step": idx % 8,
            "env": env_name,
            "attackerAction": attacker_action,
            "defenderAction": defender_action,
            "compromised": not caught,
            "taskSuccess": caught,
            "reward": row.get("reward", 0),
            "riskScore": 0.8 if attack_present else 0.2,
        })

    match_input = {
        "title": args.title,
        "attackerName": args.attacker_name,
        "defenderName": args.defender_name,
        "broadcastMode": True,
        "characterStyle": args.character_style,
        "events": events,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(match_input, f, indent=2)

    caught_count = sum(1 for e in events if e["taskSuccess"])
    missed_count = sum(1 for e in events if e["compromised"])
    print(f"Wrote {len(events)} fight events to {out_path}")
    print(f"  Caught: {caught_count}, Missed: {missed_count}")
    print(f"  Episodes: {episode_num}, Character style: {args.character_style}")


if __name__ == "__main__":
    main()
