from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

from datasets import load_dataset
from peft import LoraConfig
from trl import GRPOConfig, GRPOTrainer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agentforge_env.reward import compute_reward, invalid_action_reward, RewardConfig
from agentforge_env.serialization import (
    extract_first_json_object,
    parse_oversight_response,
    repair_action_dict,
    safe_json_load,
    ACTION_FIELDS,
)
from agentforge_env.tracks import load_seed_episodes

logger = logging.getLogger(__name__)

QWEN_ENDOFTEXT_ID = 151643
QWEN_IM_END_ID = 151645

RL_REWARD_CONFIG = RewardConfig(
    goal_degradation_tolerance=0.30,
    violation_types_points=3.0,
    culprit_localization_points=4.0,
    root_cause_points=3.0,
    recommended_action_points=2.0,
    invalid_action_penalty=-1.0,
    false_alarm_penalty_clean=-1.5,
    missed_attack_penalty=-1.5,
    missed_failure_penalty=-0.5,
    severe_risk_undercall_penalty=-0.5,
    non_json_only_penalty=-0.5,
    reward_min=-3.0,
    reward_max=20.0,
)


def _completion_to_text(item: Any) -> str:
    if isinstance(item, str):
        return item
    if isinstance(item, list):
        parts = []
        for msg in item:
            if isinstance(msg, dict) and isinstance(msg.get("content"), str):
                parts.append(msg["content"])
        return "\n".join(parts)
    return str(item)


def _score_partial_json(payload: dict[str, Any], gold: dict[str, Any]) -> float:
    """Graduate reward for JSON that parses but fails strict schema validation.

    Returns a score in [-1.0, +4.0] so the model has a gradient between
    'total garbage' and 'valid schema'.
    """
    score = 1.0  # base credit for producing parseable JSON

    present_fields = set(payload.keys()) & set(ACTION_FIELDS.keys())
    score += min(1.0, len(present_fields) / len(ACTION_FIELDS))

    if "attack_detected" in payload:
        pred = payload["attack_detected"]
        gold_val = gold.get("attack_present", False)
        if isinstance(pred, bool) and pred == bool(gold_val):
            score += 1.5

    if "risk_level" in payload and isinstance(payload["risk_level"], str):
        score += 0.5

    return score


def build_reward_func(episodes_by_id: dict[str, dict[str, Any]], *, reward_repair: bool):
    config = RL_REWARD_CONFIG

    def _tail_penalty(text: str) -> float:
        json_blob = extract_first_json_object(text)
        if not json_blob:
            return 0.0
        end = text.find(json_blob) + len(json_blob)
        tail = text[end:].strip()
        if not tail:
            return 0.0
        return -min(1.0, len(tail) / 200.0)

    def reward_func(
        prompts: list[Any],
        completions: list[Any],
        completion_ids: list[Any],  # noqa: ARG001
        episode_id: list[str] | None = None,
        **kwargs: Any,  # noqa: ARG001
    ) -> list[float]:
        rewards: list[float] = []
        ids = episode_id or []
        for idx, completion in enumerate(completions):
            if idx >= len(ids):
                rewards.append(float(invalid_action_reward("Missing episode_id.", config)["total_reward"]))
                continue
            ep_id = ids[idx]
            episode = episodes_by_id.get(ep_id)
            if episode is None:
                rewards.append(float(invalid_action_reward(f"Unknown episode_id={ep_id}.", config)["total_reward"]))
                continue

            text = _completion_to_text(completion)

            json_blob = extract_first_json_object(text)
            if json_blob is None:
                rewards.append(float(invalid_action_reward("No JSON found.", config)["total_reward"]))
                continue

            payload = safe_json_load(json_blob)
            if payload is None:
                rewards.append(-0.5)
                continue

            action, parse_meta = parse_oversight_response(text)
            if action is None or not parse_meta["schema_valid"]:
                partial = _score_partial_json(payload, episode["ground_truth"])
                rewards.append(partial + _tail_penalty(text))
                continue

            scored_action = action
            if reward_repair:
                scored_action = repair_action_dict(action, prompts[idx], calibration="conservative")
            result = compute_reward(scored_action, episode["ground_truth"], parse_meta=parse_meta, config=config)
            format_bonus = 2.0
            reward = float(result["total_reward"]) + format_bonus + _tail_penalty(text)
            rewards.append(reward)
        return rewards

    return reward_func


def _to_prompt_row(example: dict[str, Any]) -> dict[str, Any]:
    prompt = example["prompt"] + "\n\nJSON:\n"
    return {
        "prompt": prompt,
        "episode_id": example["episode_id"],
        "track": example.get("track", ""),
        "difficulty": example.get("difficulty", ""),
        "attack_family": example.get("attack_family", ""),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--train_file", type=str, default=str(ROOT / "generated_dataset_banking" / "train.jsonl"))
    parser.add_argument("--dev_file", type=str, default=str(ROOT / "generated_dataset_banking" / "dev.jsonl"))
    parser.add_argument("--episodes", type=str, default=str(ROOT / "data" / "banking_episodes_249.json"))
    parser.add_argument("--schema", type=str, default=str(ROOT / "data" / "schema.json"))
    parser.add_argument("--output_dir", type=str, default=str(ROOT / "outputs" / "rl_grpo_05b_banking"))
    parser.add_argument("--max_prompt_length", type=int, default=3072)
    parser.add_argument("--max_completion_length", type=int, default=512)
    parser.add_argument("--num_generations", type=int, default=8)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=5e-6)
    parser.add_argument("--num_train_epochs", type=float, default=1.0)
    parser.add_argument("--max_steps", type=int, default=-1)
    parser.add_argument("--logging_steps", type=int, default=1)
    parser.add_argument("--eval_steps", type=int, default=25)
    parser.add_argument("--save_steps", type=int, default=25)
    parser.add_argument("--save_total_limit", type=int, default=3)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=0.04)
    parser.add_argument("--use_vllm", action="store_true")
    parser.add_argument("--vllm_server_base_url", type=str, default=None)
    parser.add_argument("--report_to", type=str, default="none")
    parser.add_argument("--reward_repair", action="store_true")
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    args = parser.parse_args()

    try:
        episodes = load_seed_episodes(args.episodes, schema_path=args.schema)
    except Exception:
        with open(args.episodes, "r", encoding="utf-8") as f:
            episodes = json.load(f)
    episodes_by_id = {ep["episode_id"]: ep for ep in episodes}
    logger.info("Loaded %d episodes from %s", len(episodes_by_id), args.episodes)

    train_ds = load_dataset("json", data_files=args.train_file)["train"].map(_to_prompt_row)
    eval_ds = load_dataset("json", data_files=args.dev_file)["train"].map(_to_prompt_row)

    reward_func = build_reward_func(episodes_by_id, reward_repair=args.reward_repair)

    eos_ids = [QWEN_IM_END_ID, QWEN_ENDOFTEXT_ID]
    grpo_args = GRPOConfig(
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        max_prompt_length=args.max_prompt_length,
        max_completion_length=args.max_completion_length,
        num_generations=args.num_generations,
        logging_steps=args.logging_steps,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        bf16=True,
        gradient_checkpointing=True,
        remove_unused_columns=False,
        temperature=args.temperature,
        top_p=args.top_p,
        beta=args.beta,
        use_vllm=args.use_vllm,
        vllm_mode="server",
        vllm_server_base_url=args.vllm_server_base_url,
        report_to=args.report_to,
        log_completions=True,
        generation_kwargs={"eos_token_id": eos_ids},
    )

    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )

    trainer = GRPOTrainer(
        model=args.model_name,
        reward_funcs=[reward_func],
        args=grpo_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        peft_config=peft_config,
    )

    # Patch eos_token_id so TRL's post-generation EOS masking recognises both
    # <|endoftext|> (raw-text SFT stop) and <|im_end|> (chat stop).
    trainer.eos_token_id = QWEN_ENDOFTEXT_ID
    logger.info("EOS token IDs for generation: %s, masking eos_token_id: %d", eos_ids, trainer.eos_token_id)

    trainer.train()
    trainer.save_model(args.output_dir)

    run_config_path = Path(args.output_dir) / "rl_run_config.json"
    run_config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(run_config_path, "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)


if __name__ == "__main__":
    main()
