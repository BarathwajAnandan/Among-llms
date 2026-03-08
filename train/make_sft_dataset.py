from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agentforge_env.serialization import serialize_episode_for_oversight
from agentforge_env.tracks import load_seed_episodes

FORMAT_VARIANTS = [
    "Return one JSON object only. No prose before or after.",
    "Output must be valid JSON and nothing else.",
    "Do not wrap the JSON in markdown fences or commentary.",
]


def build_prompt_variants(prompt: str, approved_remediations: list[str]) -> list[str]:
    variants = [prompt]
    for idx, instruction in enumerate(FORMAT_VARIANTS, start=1):
        variant = (
            f"{prompt}\n"
            f"\nFORMAT_REMINDER_{idx}: {instruction}\n"
            "Set attack_detected to true only when an attack attempt exists.\n"
            f"recommended_action must match one of: {json.dumps(approved_remediations, ensure_ascii=False)}"
        )
        variants.append(variant)
    return variants


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=str, required=True)
    parser.add_argument("--schema", type=str, default=str(ROOT / "data" / "schema.json"))
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    episodes = load_seed_episodes(args.episodes, schema_path=args.schema)

    rng = random.Random(args.seed)
    ordered = list(episodes)
    rng.shuffle(ordered)

    n = len(ordered)
    n_train = max(1, int(0.7 * n))
    n_dev = max(1, int(0.1 * n))
    train_eps = ordered[:n_train]
    dev_eps = ordered[n_train : n_train + n_dev]
    test_eps = ordered[n_train + n_dev :]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    def write_split(name: str, split: list[dict]) -> None:
        path = out_dir / f"{name}.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            for ep in split:
                prompt = serialize_episode_for_oversight(ep)
                prompts = (
                    build_prompt_variants(prompt, ep["ground_truth"]["approved_remediations"])
                    if name == "train"
                    else [prompt]
                )
                for variant_idx, prompt_variant in enumerate(prompts):
                    row = {
                        "prompt": prompt_variant,
                        "completion": json.dumps(ep["oversight_target"], ensure_ascii=False, sort_keys=False),
                        "episode_id": ep["episode_id"],
                        "track": ep["track"],
                        "difficulty": ep["difficulty"],
                        "attack_family": ep["attack_family"],
                        "prompt_variant": variant_idx,
                    }
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")

    write_split("train", train_eps)
    write_split("dev", dev_eps)
    write_split("test", test_eps)

    manifest = {
        "seed": args.seed,
        "count": n,
        "episodes_per_split": {"train": len(train_eps), "dev": len(dev_eps), "test": len(test_eps)},
        "rows_per_split": {
            "train": len(train_eps) * (len(FORMAT_VARIANTS) + 1),
            "dev": len(dev_eps),
            "test": len(test_eps),
        },
    }
    with open(out_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"Wrote dataset to {out_dir}")


if __name__ == "__main__":
    main()
