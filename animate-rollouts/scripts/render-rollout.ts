import {mkdir, readFile} from 'node:fs/promises';
import {spawn} from 'node:child_process';
import path from 'node:path';
import {parseRolloutLog} from '../src/lib/log-parser';
import {compileMatch} from '../src/lib/timeline';
import type {CharacterStyle, MatchInput} from '../src/types';

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

const getBoolean = (args: CliArgs, key: string, fallback: boolean): boolean => {
  const value = args[key];

  if (typeof value === 'boolean') {
    return value;
  }

  if (typeof value === 'string') {
    const normalized = value.trim().toLowerCase();
    if (normalized === 'true' || normalized === '1' || normalized === 'yes' || normalized === 'on') {
      return true;
    }

    if (normalized === 'false' || normalized === '0' || normalized === 'no' || normalized === 'off') {
      return false;
    }
  }

  return fallback;
};

const loadInput = async (args: CliArgs): Promise<MatchInput> => {
  const propsPath = args.props;
  if (typeof propsPath === 'string') {
    const resolved = path.resolve(propsPath);
    const raw = await readFile(resolved, 'utf8');
    return JSON.parse(raw) as MatchInput;
  }

  const logPath = path.resolve(getString(args, 'log', 'logs/sample-rollout.log'));
  const raw = await readFile(logPath, 'utf8');
  const events = parseRolloutLog(raw);

  if (events.length === 0) {
    throw new Error(`No events parsed from ${logPath}.`);
  }

  const styleArg = getString(args, 'character-style', 'sprite');
  const characterStyle: CharacterStyle = styleArg === 'sprite' ? 'sprite' : 'among-us';

  return {
    title: getString(args, 'title', 'Attacker vs Defender: Rollout Rumble'),
    attackerName: getString(args, 'attacker', 'Environment Attacker'),
    defenderName: getString(args, 'defender', 'Task Defender'),
    broadcastMode: getBoolean(args, 'broadcast', false),
    characterStyle,
    events,
  };
};

const main = async () => {
  const args = parseArgs(process.argv.slice(2));
  const outPath = path.resolve(getString(args, 'out', 'out/rollout-match.mp4'));
  const input = await loadInput(args);
  const compiled = compileMatch(input);

  await mkdir(path.dirname(outPath), {recursive: true});

  console.log(`Rendering ${compiled.rounds.length} rounds to ${outPath}`);
  console.log(`Duration: ${(compiled.totalFrames / 30).toFixed(2)} seconds`);

  const runner = process.platform === 'win32' ? 'npx.cmd' : 'npx';
  const renderArgs = [
    'remotion',
    'render',
    'src/index.ts',
    'RolloutWrestling',
    outPath,
    '--props',
    JSON.stringify(input),
    '--overwrite',
  ];

  const child = spawn(runner, renderArgs, {
    stdio: 'inherit',
    cwd: process.cwd(),
  });

  child.on('exit', (code) => {
    process.exit(code ?? 1);
  });
};

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
