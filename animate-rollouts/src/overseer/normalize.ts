import {readFile} from 'node:fs/promises';
import type {EpisodeResult, OverseerVizInput, RunMetrics, RunSummary, TrackSummary} from './types';

type UnknownRecord = Record<string, unknown>;

export type RunBuildSpec = {
  name: string;
  metricsPath: string;
  predictionsPath?: string;
};

type SeedEpisode = {
  episodeId: string;
  attackerGoal?: string;
  defenderBehaviorSummary?: string;
  groundTruth?: string;
  oversightTarget?: string;
};

const asRecord = (value: unknown): UnknownRecord | null => {
  if (value === null || typeof value !== 'object' || Array.isArray(value)) {
    return null;
  }

  return value as UnknownRecord;
};

const pick = (record: UnknownRecord, keys: string[]): unknown => {
  for (const key of keys) {
    if (key in record) {
      return record[key];
    }
  }

  return undefined;
};

const toNumber = (value: unknown, fallback = 0): number => {
  if (typeof value === 'number' && Number.isFinite(value)) {
    return value;
  }

  const parsed = Number.parseFloat(String(value ?? ''));
  return Number.isFinite(parsed) ? parsed : fallback;
};

const toBoolean = (value: unknown): boolean | undefined => {
  if (typeof value === 'boolean') {
    return value;
  }

  if (typeof value === 'number') {
    if (value === 1) {
      return true;
    }

    if (value === 0) {
      return false;
    }
  }

  if (typeof value === 'string') {
    const normalized = value.trim().toLowerCase();
    if (normalized === 'true' || normalized === '1' || normalized === 'yes') {
      return true;
    }

    if (normalized === 'false' || normalized === '0' || normalized === 'no') {
      return false;
    }
  }

  return undefined;
};

const toString = (value: unknown, fallback = ''): string => {
  if (typeof value === 'string') {
    return value;
  }

  if (value === undefined || value === null) {
    return fallback;
  }

  return String(value);
};

const normalizeEpisodeId = (value: unknown, fallback: string): string => {
  const asText = toString(value, fallback).trim();
  return asText.length > 0 ? asText : fallback;
};

const normalizeRewardComponents = (value: unknown): Record<string, number> | undefined => {
  const record = asRecord(value);
  if (!record) {
    return undefined;
  }

  const normalized: Record<string, number> = {};
  for (const [key, raw] of Object.entries(record)) {
    const parsed = toNumber(raw, Number.NaN);
    if (Number.isFinite(parsed)) {
      normalized[key] = parsed;
    }
  }

  return Object.keys(normalized).length > 0 ? normalized : undefined;
};

const normalizePredictionObject = (value: unknown): Record<string, unknown> | undefined => {
  if (value === undefined || value === null) {
    return undefined;
  }

  const direct = asRecord(value);
  if (direct) {
    return direct;
  }

  if (typeof value === 'string') {
    try {
      const parsed = JSON.parse(value) as unknown;
      const parsedRecord = asRecord(parsed);
      return parsedRecord ?? undefined;
    } catch {
      return undefined;
    }
  }

  return undefined;
};

const normalizeTrackSummary = (value: unknown): Record<string, TrackSummary> | undefined => {
  const record = asRecord(value);
  if (!record) {
    return undefined;
  }

  const normalized: Record<string, TrackSummary> = {};
  for (const [track, rawTrackMetrics] of Object.entries(record)) {
    const trackMetrics = asRecord(rawTrackMetrics);
    if (!trackMetrics) {
      continue;
    }

    normalized[track] = {
      count: toNumber(pick(trackMetrics, ['count']), 0),
      meanReward: toNumber(pick(trackMetrics, ['mean_reward', 'meanReward']), 0),
    };
  }

  return Object.keys(normalized).length > 0 ? normalized : undefined;
};

const normalizeMetrics = (raw: UnknownRecord): RunMetrics => {
  return {
    count: toNumber(pick(raw, ['count']), 0),
    attackAccuracy: toNumber(pick(raw, ['attack_accuracy', 'attackAccuracy']), Number.NaN),
    failureAccuracy: toNumber(pick(raw, ['failure_accuracy', 'failureAccuracy']), Number.NaN),
    meanReward: toNumber(pick(raw, ['mean_reward', 'meanReward']), Number.NaN),
    falsePositiveRate: toNumber(
      pick(raw, ['false_positive_rate', 'falsePositiveRate']),
      Number.NaN,
    ),
    culpritExactRate: toNumber(pick(raw, ['culprit_exact_rate', 'culpritExactRate']), Number.NaN),
    invalidActionRate: toNumber(pick(raw, ['invalid_action_rate', 'invalidActionRate']), Number.NaN),
    jsonFoundRate: toNumber(pick(raw, ['json_found_rate', 'jsonFoundRate']), Number.NaN),
    jsonOnlyRate: toNumber(pick(raw, ['json_only_rate', 'jsonOnlyRate']), Number.NaN),
    schemaValidRate: toNumber(pick(raw, ['schema_valid_rate', 'schemaValidRate']), Number.NaN),
    tracks: normalizeTrackSummary(pick(raw, ['tracks'])),
  };
};

const sanitizeMetrics = (metrics: RunMetrics): RunMetrics => {
  const cleaned: RunMetrics = {
    count: metrics.count,
  };

  if (metrics.attackAccuracy !== undefined && Number.isFinite(metrics.attackAccuracy)) {
    cleaned.attackAccuracy = metrics.attackAccuracy;
  }

  if (metrics.failureAccuracy !== undefined && Number.isFinite(metrics.failureAccuracy)) {
    cleaned.failureAccuracy = metrics.failureAccuracy;
  }

  if (metrics.meanReward !== undefined && Number.isFinite(metrics.meanReward)) {
    cleaned.meanReward = metrics.meanReward;
  }

  if (metrics.falsePositiveRate !== undefined && Number.isFinite(metrics.falsePositiveRate)) {
    cleaned.falsePositiveRate = metrics.falsePositiveRate;
  }

  if (metrics.culpritExactRate !== undefined && Number.isFinite(metrics.culpritExactRate)) {
    cleaned.culpritExactRate = metrics.culpritExactRate;
  }

  if (metrics.invalidActionRate !== undefined && Number.isFinite(metrics.invalidActionRate)) {
    cleaned.invalidActionRate = metrics.invalidActionRate;
  }

  if (metrics.jsonFoundRate !== undefined && Number.isFinite(metrics.jsonFoundRate)) {
    cleaned.jsonFoundRate = metrics.jsonFoundRate;
  }

  if (metrics.jsonOnlyRate !== undefined && Number.isFinite(metrics.jsonOnlyRate)) {
    cleaned.jsonOnlyRate = metrics.jsonOnlyRate;
  }

  if (metrics.schemaValidRate !== undefined && Number.isFinite(metrics.schemaValidRate)) {
    cleaned.schemaValidRate = metrics.schemaValidRate;
  }

  if (metrics.tracks) {
    cleaned.tracks = metrics.tracks;
  }

  return cleaned;
};

const normalizeEpisodeRow = (
  row: UnknownRecord,
  index: number,
  seedEpisode?: SeedEpisode,
): EpisodeResult => {
  const episodeId = normalizeEpisodeId(
    pick(row, ['episode_id', 'episodeId', 'id']),
    `episode-${index + 1}`,
  );

  return {
    episodeId,
    track: toString(pick(row, ['track']), 'unknown'),
    difficulty: Math.max(0, Math.floor(toNumber(pick(row, ['difficulty']), 0))),
    attackFamily: toString(pick(row, ['attack_family', 'attackFamily']), 'unknown'),
    reward: toNumber(pick(row, ['reward']), 0),
    prediction: normalizePredictionObject(pick(row, ['prediction', 'predicted_json', 'verdict'])),
    rewardComponents: normalizeRewardComponents(
      pick(row, ['reward_components', 'rewardComponents']),
    ),
    flags: {
      attackCorrect: toBoolean(pick(row, ['attack_correct', 'attackCorrect'])),
      failureCorrect: toBoolean(pick(row, ['failure_correct', 'failureCorrect'])),
      culpritExact: toBoolean(pick(row, ['culprit_exact', 'culpritExact'])),
      falsePositive: toBoolean(pick(row, ['false_positive', 'falsePositive'])),
      invalidAction: toBoolean(pick(row, ['invalid_action', 'invalidAction'])),
      jsonFound: toBoolean(pick(row, ['json_found', 'jsonFound'])),
      jsonOnly: toBoolean(pick(row, ['json_only', 'jsonOnly'])),
      schemaValid: toBoolean(pick(row, ['schema_valid', 'schemaValid'])),
    },
    attackerGoal: seedEpisode?.attackerGoal,
    defenderBehaviorSummary: seedEpisode?.defenderBehaviorSummary,
    groundTruth: seedEpisode?.groundTruth,
    oversightTarget: seedEpisode?.oversightTarget,
    rawOutput: toString(pick(row, ['raw_output', 'rawOutput']), ''),
    error: toString(pick(row, ['error']), ''),
  };
};

const parseJsonLines = async (filePath: string): Promise<UnknownRecord[]> => {
  const raw = await readFile(filePath, 'utf8');
  const lines = raw.split(/\r?\n/);

  const parsed: UnknownRecord[] = [];
  for (const line of lines) {
    const trimmed = line.trim();
    if (trimmed.length === 0) {
      continue;
    }

    try {
      const json = JSON.parse(trimmed) as unknown;
      const asObj = asRecord(json);
      if (asObj) {
        parsed.push(asObj);
      }
    } catch {
      // Ignore malformed prediction rows.
    }
  }

  return parsed;
};

const normalizeSeedEpisode = (record: UnknownRecord): SeedEpisode | null => {
  const episodeId = normalizeEpisodeId(
    pick(record, ['episode_id', 'episodeId', 'id']),
    '',
  );

  if (episodeId.length === 0) {
    return null;
  }

  const defender = asRecord(pick(record, ['defender']));

  return {
    episodeId,
    attackerGoal: toString(pick(record, ['attacker_goal', 'attackerGoal']), ''),
    defenderBehaviorSummary: toString(
      pick(defender ?? {}, ['actual_behavior_summary', 'actualBehaviorSummary']),
      '',
    ),
    groundTruth: toString(pick(record, ['ground_truth', 'groundTruth']), ''),
    oversightTarget: toString(pick(record, ['oversight_target', 'oversightTarget']), ''),
  };
};

const indexSeedEpisodes = (value: unknown): Map<string, SeedEpisode> => {
  const map = new Map<string, SeedEpisode>();
  const root = asRecord(value);

  let list: unknown[] = [];
  if (Array.isArray(value)) {
    list = value;
  } else if (root) {
    const episodes = pick(root, ['episodes', 'data']);
    if (Array.isArray(episodes)) {
      list = episodes;
    }
  }

  for (const item of list) {
    const asObj = asRecord(item);
    if (!asObj) {
      continue;
    }

    const normalized = normalizeSeedEpisode(asObj);
    if (normalized) {
      map.set(normalized.episodeId, normalized);
    }
  }

  return map;
};

const mergePredictionRows = (
  baseRows: EpisodeResult[],
  predictionRows: UnknownRecord[],
): EpisodeResult[] => {
  const predictionsByEpisode = new Map<string, UnknownRecord>();

  for (const row of predictionRows) {
    const episodeId = normalizeEpisodeId(
      pick(row, ['episode_id', 'episodeId', 'id']),
      '',
    );
    if (episodeId.length === 0) {
      continue;
    }

    predictionsByEpisode.set(episodeId, row);
  }

  return baseRows.map((row) => {
    const predictionRow = predictionsByEpisode.get(row.episodeId);
    if (!predictionRow) {
      return row;
    }

    const prediction =
      row.prediction ??
      normalizePredictionObject(pick(predictionRow, ['prediction', 'predicted_json', 'verdict']));
    const rewardComponents =
      row.rewardComponents ??
      normalizeRewardComponents(pick(predictionRow, ['reward_components', 'rewardComponents']));
    const rawOutput = row.rawOutput || toString(pick(predictionRow, ['raw_output', 'rawOutput']), '');
    const error = row.error || toString(pick(predictionRow, ['error']), '');

    return {
      ...row,
      prediction,
      rewardComponents,
      rawOutput,
      error,
    };
  });
};

const sortEpisodes = (episodes: EpisodeResult[]): EpisodeResult[] => {
  return [...episodes].sort((a, b) => a.episodeId.localeCompare(b.episodeId, undefined, {numeric: true}));
};

export const loadSeedIndex = async (seedPath?: string): Promise<Map<string, SeedEpisode>> => {
  if (!seedPath) {
    return new Map();
  }

  const raw = await readFile(seedPath, 'utf8');
  const parsed = JSON.parse(raw) as unknown;
  return indexSeedEpisodes(parsed);
};

export const loadRunSummary = async (
  spec: RunBuildSpec,
  seedIndex: Map<string, SeedEpisode>,
): Promise<RunSummary> => {
  const rawMetricsFile = await readFile(spec.metricsPath, 'utf8');
  const metricsFile = JSON.parse(rawMetricsFile) as unknown;
  const metricsRecord = asRecord(metricsFile);

  if (!metricsRecord) {
    throw new Error(`Invalid metrics JSON object: ${spec.metricsPath}`);
  }

  const metricsRawRecord = asRecord(pick(metricsRecord, ['metrics'])) ?? {};
  const metrics = sanitizeMetrics(normalizeMetrics(metricsRawRecord));

  const rawPerEpisodes = pick(metricsRecord, ['per_episode', 'perEpisode']);
  const normalizedRows: EpisodeResult[] = Array.isArray(rawPerEpisodes)
    ? rawPerEpisodes
        .map((row, index) => {
          const asObj = asRecord(row);
          if (!asObj) {
            return null;
          }

          const episodeId = normalizeEpisodeId(
            pick(asObj, ['episode_id', 'episodeId', 'id']),
            `episode-${index + 1}`,
          );
          return normalizeEpisodeRow(asObj, index, seedIndex.get(episodeId));
        })
        .filter((row): row is EpisodeResult => row !== null)
    : [];

  const predictionRows = spec.predictionsPath
    ? await parseJsonLines(spec.predictionsPath)
    : [];

  const withPredictions = mergePredictionRows(normalizedRows, predictionRows);

  return {
    name: spec.name,
    mode: toString(pick(metricsRecord, ['mode']), ''),
    runId: toString(pick(metricsRecord, ['run_id', 'runId']), ''),
    metrics,
    episodes: sortEpisodes(withPredictions),
  };
};

export const buildOverseerVizInput = (params: {
  title: string;
  subtitle?: string;
  selectedEpisodeId?: string;
  preferredRunName?: string;
  runs: RunSummary[];
}): OverseerVizInput => {
  return {
    title: params.title,
    subtitle: params.subtitle,
    selectedEpisodeId: params.selectedEpisodeId,
    preferredRunName: params.preferredRunName,
    runs: params.runs,
  };
};
