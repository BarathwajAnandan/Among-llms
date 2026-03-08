import type {RolloutEvent} from '../types';

type RawRecord = Record<string, unknown>;

const toBoolean = (value: unknown): boolean => {
  if (typeof value === 'boolean') {
    return value;
  }

  const normalized = String(value ?? '')
    .trim()
    .toLowerCase();

  return normalized === 'true' || normalized === '1' || normalized === 'yes';
};

const toNumber = (value: unknown, fallback: number): number => {
  if (typeof value === 'number' && Number.isFinite(value)) {
    return value;
  }

  const parsed = Number.parseFloat(String(value ?? ''));
  return Number.isFinite(parsed) ? parsed : fallback;
};

const pick = (record: RawRecord, keys: string[]): unknown => {
  for (const key of keys) {
    if (key in record) {
      return record[key];
    }
  }

  return undefined;
};

const fromKvLine = (line: string): RawRecord => {
  const tokens = line
    .split(/\s+/)
    .map((token) => token.trim())
    .filter(Boolean);

  const parsed: RawRecord = {};

  for (const token of tokens) {
    const separator = token.indexOf('=');
    if (separator <= 0) {
      continue;
    }

    const key = token.slice(0, separator).trim();
    const value = token.slice(separator + 1).trim().replace(/,+$/, '');
    if (key.length === 0) {
      continue;
    }

    parsed[key] = value;
  }

  return parsed;
};

const normalizeRecord = (record: RawRecord, fallbackStep: number): RolloutEvent => {
  const episode = Math.max(1, Math.floor(toNumber(pick(record, ['episode', 'ep']), 1)));
  const step = Math.max(0, Math.floor(toNumber(pick(record, ['step', 'timestep', 'tick']), fallbackStep)));
  const env = String(pick(record, ['env', 'environment', 'arena']) ?? 'Unknown-Arena');
  const attackerAction = String(
    pick(record, ['attacker_action', 'attackerAction', 'attacker', 'attack']) ?? 'probe',
  );
  const defenderAction = String(
    pick(record, ['defender_action', 'defenderAction', 'defender', 'defense']) ?? 'stabilize',
  );

  const compromised = toBoolean(
    pick(record, ['compromised', 'is_compromised', 'defender_compromised']) ?? false,
  );
  const taskSuccess = toBoolean(
    pick(record, ['task_success', 'taskSuccess', 'success', 'objective_complete']) ?? false,
  );

  const reward = pick(record, ['reward', 'r']);
  const riskScore = pick(record, ['risk_score', 'risk', 'threat']);
  const timestamp = pick(record, ['timestamp', 'time']);

  const event: RolloutEvent = {
    episode,
    step,
    env,
    attackerAction,
    defenderAction,
    compromised,
    taskSuccess,
  };

  if (reward !== undefined) {
    event.reward = toNumber(reward, 0);
  }

  if (riskScore !== undefined) {
    event.riskScore = toNumber(riskScore, 0);
  }

  if (timestamp !== undefined) {
    event.timestamp = String(timestamp);
  }

  return event;
};

const parseLine = (line: string): RawRecord | null => {
  const trimmed = line.trim();
  if (trimmed.length === 0 || trimmed.startsWith('#')) {
    return null;
  }

  if (trimmed.startsWith('{')) {
    try {
      return JSON.parse(trimmed) as RawRecord;
    } catch {
      return null;
    }
  }

  const kvParsed = fromKvLine(trimmed);
  if (Object.keys(kvParsed).length === 0) {
    return null;
  }

  return kvParsed;
};

export const parseRolloutLog = (rawLog: string): RolloutEvent[] => {
  const lines = rawLog.split(/\r?\n/);

  const events = lines
    .map((line, index) => {
      const parsed = parseLine(line);
      if (parsed === null) {
        return null;
      }

      return normalizeRecord(parsed, index);
    })
    .filter((event): event is RolloutEvent => event !== null);

  return events.sort((a, b) => {
    if (a.episode !== b.episode) {
      return a.episode - b.episode;
    }

    return a.step - b.step;
  });
};
