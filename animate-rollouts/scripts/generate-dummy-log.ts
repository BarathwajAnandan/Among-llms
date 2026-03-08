import {mkdir, writeFile} from 'node:fs/promises';
import path from 'node:path';

type CliArgs = Record<string, string | boolean>;

const parseArgs = (argv: string[]): CliArgs => {
  const result: CliArgs = {};

  for (let index = 0; index < argv.length; index += 1) {
    const arg = argv[index];
    if (!arg.startsWith('--')) {
      continue;
    }

    const trimmed = arg.slice(2);
    const separator = trimmed.indexOf('=');

    if (separator >= 0) {
      result[trimmed.slice(0, separator)] = trimmed.slice(separator + 1);
      continue;
    }

    const next = argv[index + 1];
    if (next && !next.startsWith('--')) {
      result[trimmed] = next;
      index += 1;
      continue;
    }

    result[trimmed] = true;
  }

  return result;
};

const getString = (args: CliArgs, key: string, fallback: string): string => {
  const value = args[key];
  return typeof value === 'string' ? value : fallback;
};

const getNumber = (args: CliArgs, key: string, fallback: number): number => {
  const value = args[key];
  if (typeof value !== 'string') {
    return fallback;
  }

  const parsed = Number.parseInt(value, 10);
  return Number.isFinite(parsed) ? parsed : fallback;
};

const hashSeed = (seed: string): number => {
  let result = 0;
  for (let index = 0; index < seed.length; index += 1) {
    result = (result << 5) - result + seed.charCodeAt(index);
    result |= 0;
  }

  return Math.abs(result) || 1;
};

const makeRng = (seed: string) => {
  let state = hashSeed(seed);
  return () => {
    state = (1664525 * state + 1013904223) % 4294967296;
    return state / 4294967296;
  };
};

const environments = [
  'NeonGrid-001',
  'BreachFactory-02',
  'CitadelCore-77',
  'SandstormVault-12',
  'NeonGrid-Redux',
  'DeepLabyrinth-5',
];

const attackerActions = [
  'spawn_phishing_trap',
  'inject_noise',
  'escalate_pressure',
  'trigger_decoy_alarm',
  'privilege_bait',
  'payload_drop',
  'mutate_layout',
  'stealth_probe',
  'zero_day_feint',
];

const defenderActions = [
  'scan_surface',
  'replan_route',
  'sandbox_probe',
  'verify_signal',
  'least_privilege_mode',
  'containment_lock',
  'adaptive_policy',
  'integrity_audit',
  'hot_patch',
  'fallback_route',
  'quarantine_module',
];

const main = async () => {
  const args = parseArgs(process.argv.slice(2));
  const episodes = Math.max(1, getNumber(args, 'episodes', 4));
  const stepsPerEpisode = Math.max(1, getNumber(args, 'steps', 6));
  const seed = getString(args, 'seed', 'wrestling-demo');
  const outPath = path.resolve(getString(args, 'out', 'logs/generated-dummy-rollout.log'));

  const rng = makeRng(seed);
  const lines: string[] = [];

  for (let episode = 1; episode <= episodes; episode += 1) {
    const env = environments[(episode - 1) % environments.length] ?? environments[0] ?? 'Arena-1';

    for (let step = 0; step < stepsPerEpisode; step += 1) {
      const attackerAction =
        attackerActions[Math.floor(rng() * attackerActions.length)] ?? attackerActions[0] ?? 'probe';
      const defenderAction =
        defenderActions[Math.floor(rng() * defenderActions.length)] ?? defenderActions[0] ?? 'stabilize';

      const compromised = rng() < 0.28;
      const taskSuccess = !compromised && rng() < 0.72;
      const reward = taskSuccess ? 0.2 + rng() * 0.7 : -(0.2 + rng() * 0.7);
      const riskScore = compromised ? 0.6 + rng() * 0.35 : 0.08 + rng() * 0.45;

      lines.push(
        [
          `episode=${episode}`,
          `step=${step}`,
          `env=${env}`,
          `attacker_action=${attackerAction}`,
          `defender_action=${defenderAction}`,
          `compromised=${compromised ? 'true' : 'false'}`,
          `task_success=${taskSuccess ? 'true' : 'false'}`,
          `reward=${reward.toFixed(2)}`,
          `risk_score=${riskScore.toFixed(2)}`,
        ].join(' '),
      );
    }
  }

  await mkdir(path.dirname(outPath), {recursive: true});
  await writeFile(outPath, `${lines.join('\n')}\n`, 'utf8');

  console.log(`Generated ${lines.length} events using seed '${seed}'`);
  console.log(`Output: ${outPath}`);
};

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
