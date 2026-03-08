# AgentForge Oversight

AgentForge is a one-step OpenEnv benchmark for oversight of subtly sabotaged workspace tasks.

An oversight model audits a defender agent that worked inside a digital workspace where one attacker was present through a document, message, or tool output. The overseer must detect the attack, identify the culprit source, and recommend a remediation — all in a single structured JSON action.

## Layout
- `agentforge_env/`: environment package — reward function, serialization, server, client, inference helpers
- `data/`: canonical episode schema, seed episodes, and banking episode corpus (249 episodes)
- `eval/`: evaluation scripts — baselines, metrics, parallel vLLM evaluation
- `train/`: SFT dataset builder, SFT + RL (GRPO) training scripts, reward hook

## Quick Start

```bash
pip install -e .

# Run weak baseline
python eval/run_baseline.py --episodes data/seed_episodes.json --mode weak

# Run oracle baseline
python eval/run_baseline.py --episodes data/seed_episodes.json --mode oracle

# Generate SFT dataset
python train/make_sft_dataset.py --episodes data/seed_episodes.json --out_dir generated_dataset

# Launch environment server
uvicorn app:app --host 0.0.0.0 --port 7860
```

## Training Pipeline

1. **SFT** — Teach the model JSON format and baseline detection via `train/sft_train_unsloth.py`
2. **LoRA Merge** — Merge adapter weights via `train/merge_lora_adapter.py`
3. **RL (GRPO)** — Improve attack detection via `train/rl_train_trl.py` or `train/rl_train_openenv.py`
4. **Evaluate** — Measure FN rate and reward via `eval/fast_fn_check.py` or `eval/full_eval.py`

## Hugging Face Spaces

Use the root `Dockerfile` for Docker Space deployment and `HF_SPACE_README.md` as the Space README template.
