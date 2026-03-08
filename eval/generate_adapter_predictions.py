from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agentforge_env.inference import DEFAULT_SYSTEM_PROMPT
from agentforge_env.serialization import parse_oversight_response, repair_action_dict, serialize_episode_for_oversight
from agentforge_env.tracks import load_seed_episodes


def load_base_model_name(adapter_dir: Path) -> str:
    with open(adapter_dir / "adapter_config.json", "r", encoding="utf-8") as f:
        config = json.load(f)
    return config["base_model_name_or_path"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=str, required=True)
    parser.add_argument("--adapter_dir", type=str, required=True)
    parser.add_argument("--schema", type=str, default=str(ROOT / "data" / "schema.json"))
    parser.add_argument("--out", type=str, default="adapter_predictions.jsonl")
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--load_in_4bit", action="store_true")
    parser.add_argument("--no_repair", action="store_true")
    args = parser.parse_args()

    adapter_dir = Path(args.adapter_dir)
    base_model_name = load_base_model_name(adapter_dir)
    tokenizer = AutoTokenizer.from_pretrained(adapter_dir, trust_remote_code=True)

    quantization_config = None
    model_kwargs = {
        "device_map": "auto",
        "trust_remote_code": True,
    }
    if torch.cuda.is_available():
        model_kwargs["torch_dtype"] = torch.bfloat16
    if args.load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        model_kwargs["quantization_config"] = quantization_config

    base_model = AutoModelForCausalLM.from_pretrained(base_model_name, **model_kwargs)
    model = PeftModel.from_pretrained(base_model, adapter_dir)
    model.eval()

    episodes = load_seed_episodes(args.episodes, schema_path=args.schema)
    out_path = Path(args.out)
    with open(out_path, "w", encoding="utf-8") as f:
        for ep in episodes:
            messages = [
                {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
                {"role": "user", "content": ep["scenario"].get("oversight_input", "")},
            ]
            if not messages[1]["content"]:
                messages[1]["content"] = serialize_episode_for_oversight(ep)

            input_ids = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
            ).to(model.device)
            attention_mask = torch.ones_like(input_ids)
            with torch.inference_mode():
                outputs = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.eos_token_id,
                )
            generated = outputs[0][input_ids.shape[-1] :]
            raw_output = tokenizer.decode(generated, skip_special_tokens=True).strip()
            parsed, parse_meta = parse_oversight_response(raw_output)
            if parsed is not None and parse_meta["schema_valid"] and not args.no_repair:
                parsed = repair_action_dict(parsed, messages[1]["content"])
            row = {
                "episode_id": ep["episode_id"],
                "raw_output": raw_output,
                "prediction": parsed,
                "parse_meta": parse_meta,
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(str(out_path))


if __name__ == "__main__":
    main()
