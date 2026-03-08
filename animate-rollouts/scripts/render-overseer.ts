import {readFile, writeFile, mkdtemp, rm} from 'node:fs/promises';
import {spawn} from 'node:child_process';
import os from 'node:os';
import path from 'node:path';
import {defaultOverseerVizInput} from '../src/data/default-overseer-viz';
import type {OverseerVizInput} from '../src/overseer/types';

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

const loadInput = async (args: CliArgs): Promise<OverseerVizInput> => {
  const propsPath = args.props;
  if (typeof propsPath !== 'string') {
    return defaultOverseerVizInput;
  }

  const resolved = path.resolve(propsPath);
  const raw = await readFile(resolved, 'utf8');
  return JSON.parse(raw) as OverseerVizInput;
};

const main = async () => {
  const args = parseArgs(process.argv.slice(2));
  const outPath = path.resolve(getString(args, 'out', 'out/overseer-rollout.mp4'));
  const input = await loadInput(args);

  console.log(`Rendering OverseerRolloutViz to ${outPath}`);
  console.log(`Runs: ${input.runs.map((run) => run.name).join(', ')}`);
  console.log(`Total episodes: ${input.runs.reduce((s, r) => s + r.episodes.length, 0)}`);

  const tmpDir = await mkdtemp(path.join(os.tmpdir(), 'remotion-props-'));
  const propsFile = path.join(tmpDir, 'props.json');
  await writeFile(propsFile, JSON.stringify(input), 'utf8');

  const runner = process.platform === 'win32' ? 'npx.cmd' : 'npx';
  const renderArgs = [
    'remotion',
    'render',
    'src/index.ts',
    'OverseerRolloutViz',
    outPath,
    '--props',
    propsFile,
    '--overwrite',
  ];

  const child = spawn(runner, renderArgs, {
    stdio: 'inherit',
    cwd: process.cwd(),
  });

  child.on('exit', async (code) => {
    await rm(tmpDir, {recursive: true, force: true}).catch(() => {});
    process.exit(code ?? 1);
  });
};

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
