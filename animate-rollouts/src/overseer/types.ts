export type EpisodeFlags = {
  attackCorrect?: boolean;
  failureCorrect?: boolean;
  culpritExact?: boolean;
  falsePositive?: boolean;
  invalidAction?: boolean;
  jsonFound?: boolean;
  jsonOnly?: boolean;
  schemaValid?: boolean;
};

export type EpisodeResult = {
  episodeId: string;
  track: string;
  difficulty: number;
  attackFamily: string;
  reward: number;
  prediction?: Record<string, unknown>;
  rewardComponents?: Record<string, number>;
  flags: EpisodeFlags;
  attackerGoal?: string;
  defenderBehaviorSummary?: string;
  groundTruth?: string;
  oversightTarget?: string;
  rawOutput?: string;
  error?: string;
};

export type TrackSummary = {
  count: number;
  meanReward: number;
};

export type RunMetrics = {
  count: number;
  attackAccuracy?: number;
  failureAccuracy?: number;
  meanReward?: number;
  falsePositiveRate?: number;
  culpritExactRate?: number;
  invalidActionRate?: number;
  jsonFoundRate?: number;
  jsonOnlyRate?: number;
  schemaValidRate?: number;
  tracks?: Record<string, TrackSummary>;
};

export type RunSummary = {
  name: string;
  mode?: string;
  runId?: string;
  metrics: RunMetrics;
  episodes: EpisodeResult[];
};

export type EpisodeFilters = {
  track?: string[];
  difficulty?: number[];
  attackFamily?: string[];
};

export type OverseerVizInput = {
  title: string;
  subtitle?: string;
  selectedEpisodeId?: string;
  preferredRunName?: string;
  filters?: EpisodeFilters;
  runs: RunSummary[];
};
