import {access, mkdir, writeFile} from 'node:fs/promises';
import path from 'node:path';
import {buildOverseerVizInput, loadRunSummary, loadSeedIndex} from '../src/overseer/normalize';

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

const pathExists = async (filePath: string): Promise<boolean> => {
  try {
    await access(filePath);
    return true;
  } catch {
    return false;
  }
};

const maybePath = (value: string): string | undefined => {
  const trimmed = value.trim();
  return trimmed.length > 0 ? trimmed : undefined;
};

const defaults = {
  seed: 'data/samples/overseer/seed_episodes.json',
  weakMetrics: 'data/samples/overseer/weak_metrics.json',
  dumbMetrics: 'data/samples/overseer/dumb_metrics.json',
  trainedMetrics: 'data/samples/overseer/trained_metrics.json',
  oracleMetrics: '',
};

const main = async () => {
  const args = parseArgs(process.argv.slice(2));

  const title = getString(args, 'title', 'Rollout Oversight Arena');
  const subtitle = getString(args, 'subtitle', 'Weak vs Dumb vs Trained Overseer');
  const outPath = path.resolve(getString(args, 'out', 'data/generated/overseer-viz-input.json'));
  const selectedEpisodeId = maybePath(getString(args, 'episode-id', ''));

  const seedPath = path.resolve(getString(args, 'seed', defaults.seed));
  const weakMetrics = path.resolve(getString(args, 'weak-metrics', defaults.weakMetrics));
  const weakPredictions = maybePath(getString(args, 'weak-predictions', ''));

  const dumbMetricsRaw = maybePath(getString(args, 'dumb-metrics', defaults.dumbMetrics));
  const dumbMetrics = dumbMetricsRaw ? path.resolve(dumbMetricsRaw) : '';
  const dumbPredictions = maybePath(getString(args, 'dumb-predictions', ''));

  const trainedMetricsRaw = maybePath(getString(args, 'trained-metrics', defaults.trainedMetrics));
  const trainedMetrics = trainedMetricsRaw ? path.resolve(trainedMetricsRaw) : '';
  const trainedPredictions = maybePath(getString(args, 'trained-predictions', ''));

  const oracleMetricsRaw = maybePath(getString(args, 'oracle-metrics', defaults.oracleMetrics));
  const oracleMetrics = oracleMetricsRaw ? path.resolve(oracleMetricsRaw) : '';
  const oraclePredictions = maybePath(getString(args, 'oracle-predictions', ''));

  const seedIndex = (await pathExists(seedPath)) ? await loadSeedIndex(seedPath) : new Map();

  const runSpecs = [
    {
      name: getString(args, 'weak-name', 'weak baseline'),
      metricsPath: weakMetrics,
      predictionsPath: weakPredictions,
    },
    {
      name: getString(args, 'dumb-name', 'dumb model'),
      metricsPath: dumbMetrics,
      predictionsPath: dumbPredictions,
    },
    {
      name: getString(args, 'trained-name', 'trained overseer'),
      metricsPath: trainedMetrics,
      predictionsPath: trainedPredictions,
    },
    {
      name: getString(args, 'oracle-name', 'oracle'),
      metricsPath: oracleMetrics,
      predictionsPath: oraclePredictions,
    },
  ];

  const runs = [];

  for (const spec of runSpecs) {
    if (!spec.metricsPath || !(await pathExists(spec.metricsPath))) {
      continue;
    }

    const summary = await loadRunSummary(
      {
        name: spec.name,
        metricsPath: spec.metricsPath,
        predictionsPath: spec.predictionsPath,
      },
      seedIndex,
    );
    runs.push(summary);
  }

  if (runs.length === 0) {
    throw new Error('No readable metrics files were found. Provide valid --*-metrics paths.');
  }

  const input = buildOverseerVizInput({
    title,
    subtitle,
    selectedEpisodeId,
    preferredRunName: getString(args, 'preferred-run', 'trained overseer'),
    runs,
  });

  await mkdir(path.dirname(outPath), {recursive: true});
  await writeFile(outPath, JSON.stringify(input, null, 2), 'utf8');

  console.log(`Wrote overseer visualization input to ${outPath}`);
  console.log(`Runs included: ${runs.map((run) => run.name).join(', ')}`);
  console.log(`Primary selected episode: ${selectedEpisodeId ?? '(auto)'}`);
};

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
