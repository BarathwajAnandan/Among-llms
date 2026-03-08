# AgentForge Oversight — Rollout Animations

Animated visualizations of oversight agent eval rollouts. Two compositions:

1. **RolloutWrestling** — Among Us-style fighting animation where each episode is a fight exchange (attacker lands a hit = attack missed, defender blocks = attack caught)
2. **OverseerRolloutViz** — Scoreboard/metrics comparison across model runs

Built with [Remotion](https://remotion.dev).

## Quick start

```bash
cd animate-rollouts
npm install
```

## Using with AgentForge eval data

### Step 1: Convert eval outputs to animation format

**For the fight animation** (RolloutWrestling):

```bash
python scripts/convert_eval_to_fight.py \
  --precomputed ../outputs/evals/precomputed_episode_outputs.jsonl \
  --episodes ../data/banking_episodes_249.json \
  --label "0.5B-RL" \
  --max-episodes 200 \
  --out data/generated/fight-input.json
```

**For the scoreboard** (OverseerRolloutViz):

```bash
python scripts/convert_eval_to_viz.py \
  --precomputed ../outputs/evals/precomputed_episode_outputs.jsonl \
  --episodes ../data/banking_episodes_249.json \
  --out-dir data/generated

npm run build:overseer-input -- \
  --seed=../data/banking_episodes_249.json \
  --weak-metrics=data/generated/weak_metrics.json \
  --trained-metrics=data/generated/trained_metrics.json \
  --out=data/generated/overseer-viz-input.json
```

### Step 2: Play in browser (interactive)

```bash
npm run studio -- --port 3100
```

Opens Remotion Studio at `http://localhost:3100`. Select a composition and load props:

- **RolloutWrestling** → load `data/generated/fight-input.json`
- **OverseerRolloutViz** → load `data/generated/overseer-viz-input.json`

You can play, pause, scrub frame-by-frame, and inspect the animation.

### Step 3: Render to MP4 (optional)

```bash
# Fight animation
npm run render:rollout -- --props data/generated/fight-input.json --out out/fight.mp4

# Scoreboard
npm run render:overseer -- --props data/generated/overseer-viz-input.json --out out/overseer.mp4
```

## Gradio integration

The Gradio demo (`demo/app.py`) links to the Remotion Studio in the "What We Built" tab. Set the URL via environment variable:

```bash
export REMOTION_STUDIO_URL=http://localhost:3100
```

## Converter options

### convert_eval_to_fight.py

| Flag | Default | Description |
|------|---------|-------------|
| `--precomputed` | required | Path to `precomputed_episode_outputs.jsonl` |
| `--episodes` | required | Path to `banking_episodes_249.json` |
| `--label` | `0.5B-RL` | Which model label (`0.5B-base` or `0.5B-RL`) |
| `--max-episodes` | `0` (all) | Limit number of episodes |
| `--character-style` | `among-us` | `among-us` or `sprite` |
| `--title` | `Attacker vs Oversight Agent` | Title card text |
| `--out` | required | Output JSON path |

### convert_eval_to_viz.py

| Flag | Default | Description |
|------|---------|-------------|
| `--precomputed` | required | Path to `precomputed_episode_outputs.jsonl` |
| `--episodes` | required | Path to `banking_episodes_249.json` |
| `--max-episodes` | `0` (all) | Limit episodes per run |
| `--out-dir` | required | Output directory for metrics JSONs |

## How the fight mapping works

Each eval episode becomes one fight exchange:

- **Attacker action** is mapped from the episode's `attack_family` (e.g. `phishing_prompt_injection` → `spawn_phishing_trap`)
- **Defender action** depends on whether the oversight model caught the attack (caught → `containment_lock`, missed → `adaptive_policy`)
- **`compromised=true`** (attacker wins exchange) = oversight model missed the attack
- **`taskSuccess=true`** (defender wins exchange) = oversight model caught the attack

## Assets

External combat sprites and audio under `public/assets/`. See `docs/asset-license-manifest.md` for licenses.
