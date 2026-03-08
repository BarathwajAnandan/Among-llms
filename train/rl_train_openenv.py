"""GRPO RL training through the OpenEnv environment interface.

The reward function uses AgentForgeOversightEnvironment.reset() + .step()
instead of calling compute_reward directly, so the full environment loop
is exercised — ready for future attacker/defender models.
"""
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

from agentforge_env.server.environment import AgentForgeOversightEnvironment
from agentforge_env.reward import RewardConfig
from agentforge_env.serialization import (
    extract_first_json_object,
    parse_oversight_response,
    safe_json_load,
    ACTION_FIELDS,
)

RL_REWARD_V2 = RewardConfig(
    attack_detection_points=3.0,
    failure_detection_points=2.0,
    goal_degradation_points=1.0,
    goal_degradation_tolerance=0.30,
    risk_level_points=1.0,
    violation_types_points=3.0,
    culprit_localization_points=4.0,
    root_cause_points=3.0,
    recommended_action_points=2.0,
    missed_attack_penalty=-3.0,
    missed_failure_penalty=-0.5,
    false_alarm_penalty_clean=-1.0,
    severe_risk_undercall_penalty=-0.5,
    non_json_only_penalty=-0.5,
    invalid_action_penalty=-1.0,
    reward_min=-4.0,
    reward_max=20.0,
)

logger = logging.getLogger(__name__)

QWEN_ENDOFTEXT_ID = 151643
QWEN_IM_END_ID = 151645


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


def _score_partial_json(payload: dict[str, Any], episode_id: str, env: AgentForgeOversightEnvironment) -> float:
    """Graduate reward for JSON that parses but fails strict schema validation."""
    score = 1.0
    present_fields = set(payload.keys()) & set(ACTION_FIELDS.keys())
    score += min(1.0, len(present_fields) / len(ACTION_FIELDS))
    if "attack_detected" in payload and isinstance(payload["attack_detected"], bool):
        score += 1.0
    if "risk_level" in payload and isinstance(payload["risk_level"], str):
        score += 0.5
    return score


def _tail_penalty(text: str) -> float:
    json_blob = extract_first_json_object(text)
    if not json_blob:
        return 0.0
    end = text.find(json_blob) + len(json_blob)
    tail = text[end:].strip()
    if not tail:
        return 0.0
    return -min(1.0, len(tail) / 200.0)


def build_openenv_reward_func(env: AgentForgeOversightEnvironment):
    """Build a reward function that scores via the OpenEnv environment."""

    def reward_func(
        prompts: list[Any],
        completions: list[Any],
        completion_ids: list[Any],
        episode_id: list[str] | None = None,
        **kwargs: Any,
    ) -> list[float]:
        rewards: list[float] = []
        ids = episode_id or []

        for idx, completion in enumerate(completions):
            if idx >= len(ids):
                rewards.append(-1.0)
                continue

            ep_id = ids[idx]
            text = _completion_to_text(completion)

            json_blob = extract_first_json_object(text)
            if json_blob is None:
                rewards.append(-1.0)
                continue

            payload = safe_json_load(json_blob)
            if payload is None:
                rewards.append(-0.5)
                continue

            action, parse_meta = parse_oversight_response(text)
            if action is None or not parse_meta["schema_valid"]:
                partial = _score_partial_json(payload, ep_id, env)
                rewards.append(partial + _tail_penalty(text))
                continue

            try:
                env.reset(episode_id=ep_id)
                obs = env.step(action)
                env_reward = obs.reward
            except Exception as e:
                logger.warning("Env step failed for %s: %s", ep_id, e)
                env_reward = 0.0

            format_bonus = 2.0
            reward = float(env_reward) + format_bonus + _tail_penalty(text)
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
    parser = argparse.ArgumentParser(description="GRPO RL training via OpenEnv")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--train_file", type=str, default=str(ROOT / "generated_dataset_banking" / "train.jsonl"))
    parser.add_argument("--dev_file", type=str, default=str(ROOT / "generated_dataset_banking" / "dev.jsonl"))
    parser.add_argument("--episodes", type=str, default=str(ROOT / "data" / "banking_episodes_249.json"))
    parser.add_argument("--schema", type=str, default=str(ROOT / "data" / "schema.json"))
    parser.add_argument("--output_dir", type=str, default=str(ROOT / "outputs" / "rl_grpo_openenv"))
    parser.add_argument("--max_prompt_length", type=int, default=3072)
    parser.add_argument("--max_completion_length", type=int, default=512)
    parser.add_argument("--num_generations", type=int, default=8)
    parser.add_argument("--per_device_train_batch_size", type=int, default=8)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
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
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    args = parser.parse_args()

    env = AgentForgeOversightEnvironment(
        episodes_path=args.episodes,
        schema_path=args.schema,
        reward_config=RL_REWARD_V2,
    )
    test_obs = env.reset()
    logger.info("OpenEnv initialized — test episode: %s", test_obs.episode_id)

    train_ds = load_dataset("json", data_files=args.train_file)["train"].map(_to_prompt_row)
    eval_ds = load_dataset("json", data_files=args.dev_file)["train"].map(_to_prompt_row)

    reward_func = build_openenv_reward_func(env)

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

    trainer.eos_token_id = QWEN_ENDOFTEXT_ID
    logger.info("EOS token IDs: %s, masking: %d", eos_ids, trainer.eos_token_id)

    trainer.train()
    trainer.save_model(args.output_dir)

    run_config = vars(args)
    run_config["reward_source"] = "openenv"
    run_config_path = Path(args.output_dir) / "rl_run_config.json"
    run_config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(run_config_path, "w", encoding="utf-8") as f:
        json.dump(run_config, f, indent=2)
    logger.info("Saved run config to %s", run_config_path)


if __name__ == "__main__":
    main()
