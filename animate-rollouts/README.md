# RL Rollout Wrestling Visualizer

Render RL rollout logs as a 2D wrestling-style match with multiple rounds and environment changes.

The video is generated with Remotion and exported as MP4.

## Overseer rollout handoff mode

This repo now includes a second composition, `OverseerRolloutViz`, for the handoff contract:

- stable run-level metrics + per-episode schema
- three-lane episode narrative (attacker / defender / overseer)
- reward component scoreboard
- weak vs dumb vs trained comparison cards

Render with default demo data:

```bash
npm run render:overseer -- --out out/overseer-demo.mp4
```

Build input from local metrics/prediction files (defaults match the handoff paths):

```bash
npm run build:overseer-input -- --out data/generated/overseer-viz-input.json
npm run render:overseer -- --props data/generated/overseer-viz-input.json --out out/overseer-from-files.mp4
```

Override file paths when needed:

```bash
npm run build:overseer-input -- \
  --seed /path/to/seed_episodes.json \
  --weak-metrics /path/to/weak_metrics.json \
  --dumb-metrics /path/to/dumb_metrics.json \
  --dumb-predictions /path/to/dumb_predictions.jsonl \
  --trained-metrics /path/to/trained_metrics.json \
  --trained-predictions /path/to/trained_predictions.jsonl
```

Runnable sample dataset is in `data/samples/overseer/`:

```bash
npm run build:overseer-input -- \
  --seed data/samples/overseer/seed_episodes.json \
  --weak-metrics data/samples/overseer/weak_metrics.json \
  --dumb-metrics data/samples/overseer/dumb_metrics.json \
  --dumb-predictions data/samples/overseer/dumb_predictions.jsonl \
  --trained-metrics data/samples/overseer/trained_metrics.json \
  --trained-predictions data/samples/overseer/trained_predictions.jsonl \
  --out data/generated/overseer-viz-sample.json
```

## What this project does

- Parses rollout logs (key-value lines or JSON lines).
- Maps attacker/defender events to animated wrestling exchanges.
- Uses real external sprite-sheet + audio assets (OpenGameArt) with in-project license manifest.
- Splits rounds automatically whenever environment or episode changes.
- Adds deliberate hold frames (dummy delay) so each rollout beat is visible.
- Includes an optional broadcast analytics overlay (off by default).
- Defaults to a clean HUD-only view to keep action readable.
- Produces a single deterministic MP4 render.

## Quick start

```bash
npm install
npm run studio
```

To render from the included sample log:

```bash
npm run render:rollout -- --log logs/sample-rollout.log --out out/sample-rollout.mp4
```

Enable the analytics overlay only when needed:

```bash
npm run render:rollout -- --log logs/sample-rollout.log --out out/sample-rollout-broadcast.mp4 --broadcast
```

To render from a longer dummy dataset:

```bash
npm run render:rollout -- --log logs/dummy-rollout-broadcast.log --out out/dummy-broadcast.mp4
```

## Input format

### Key-value line format

```text
episode=1 step=0 env=NeonGrid-001 attacker_action=spawn_phishing_trap defender_action=scan_surface compromised=false task_success=true reward=0.40 risk_score=0.20
```

### JSON line format

```json
{"episode":1,"step":0,"env":"NeonGrid-001","attacker_action":"spawn_phishing_trap","defender_action":"scan_surface","compromised":false,"task_success":true}
```

Supported aliases include:

- `env` / `environment`
- `attacker_action` / `attackerAction`
- `defender_action` / `defenderAction`
- `task_success` / `taskSuccess` / `success`
- `compromised` / `is_compromised`

## Build an input props file

```bash
npm run build:input -- --log logs/sample-rollout.log --out data/generated/match-input.json
```

To include analytics overlay in generated props:

```bash
npm run build:input -- --log logs/sample-rollout.log --out data/generated/match-input.json --broadcast
```

You can then render directly from props:

```bash
npm run render:rollout -- --props data/generated/match-input.json --out out/from-props.mp4
```

## Generate random dummy rollout logs

```bash
npm run generate:dummy -- --episodes 5 --steps 8 --seed demo-seed --out logs/generated-dummy-rollout.log
```

Then render:

```bash
npm run render:rollout -- --log logs/generated-dummy-rollout.log --out out/generated-dummy.mp4
```

## Scene mapping

- Intro card.
- Round intro card for each environment.
- Per-step fight choreography:
  - windup
  - impact
  - recovery
  - readability hold (dummy delay)
- Inter-round transition card.
- Final verdict screen.

## Action-to-move mapping

Rollout actions are mapped to custom move archetypes to make each exchange visually distinct:

- attacker examples: `payload_drop` -> heavy breach slam, `inject_noise` -> trap feint, `stealth_probe` -> probe jab
- defender examples: `containment_lock` -> hard block, `adaptive_policy` -> footwork pivot, `integrity_audit` -> read counter

Outcome fields drive who wins the exchange:

- `compromised=true` -> attacker lands decisive hit
- `task_success=true && compromised=false` -> defender wins exchange
- otherwise -> neutral probing exchange

## Assets

This repository includes environment/UI SVG placeholders plus external combat assets under `public/assets/`.

- Source list and license details: `docs/asset-license-manifest.md`
- General free source references: `docs/asset-sources.md`
