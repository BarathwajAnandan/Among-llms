import type {EpisodeFilters, EpisodeResult, OverseerVizInput, RunSummary} from './types';

const normalize = (value: string): string => value.trim().toLowerCase();

const matchesFilters = (episode: EpisodeResult, filters?: EpisodeFilters): boolean => {
  if (!filters) {
    return true;
  }

  if (filters.track && filters.track.length > 0) {
    const allowed = new Set(filters.track.map(normalize));
    if (!allowed.has(normalize(episode.track))) {
      return false;
    }
  }

  if (filters.attackFamily && filters.attackFamily.length > 0) {
    const allowed = new Set(filters.attackFamily.map(normalize));
    if (!allowed.has(normalize(episode.attackFamily))) {
      return false;
    }
  }

  if (filters.difficulty && filters.difficulty.length > 0) {
    const allowed = new Set(filters.difficulty);
    if (!allowed.has(episode.difficulty)) {
      return false;
    }
  }

  return true;
};

export const filterRunEpisodes = (run: RunSummary, filters?: EpisodeFilters): RunSummary => {
  const episodes = run.episodes.filter((episode) => matchesFilters(episode, filters));
  return {
    ...run,
    episodes,
  };
};

export const pickPrimaryRun = (runs: RunSummary[], preferredRunName?: string): RunSummary | null => {
  if (runs.length === 0) {
    return null;
  }

  if (preferredRunName) {
    const preferred = runs.find((run) => normalize(run.name) === normalize(preferredRunName));
    if (preferred) {
      return preferred;
    }
  }

  const trained = runs.find((run) => normalize(run.name).includes('trained'));
  return trained ?? runs[0] ?? null;
};

export const pickEpisodeId = (
  runs: RunSummary[],
  selectedEpisodeId?: string,
  preferredRunName?: string,
): string | null => {
  if (runs.length === 0) {
    return null;
  }

  if (selectedEpisodeId) {
    const exists = runs.some((run) => run.episodes.some((episode) => episode.episodeId === selectedEpisodeId));
    if (exists) {
      return selectedEpisodeId;
    }
  }

  const primary = pickPrimaryRun(runs, preferredRunName);
  if (primary && primary.episodes[0]) {
    return primary.episodes[0].episodeId;
  }

  const firstRunWithEpisodes = runs.find((run) => run.episodes.length > 0);
  return firstRunWithEpisodes?.episodes[0]?.episodeId ?? null;
};

export const getEpisodeForRun = (run: RunSummary, episodeId: string | null): EpisodeResult | null => {
  if (!episodeId) {
    return run.episodes[0] ?? null;
  }

  return run.episodes.find((episode) => episode.episodeId === episodeId) ?? run.episodes[0] ?? null;
};

export const resolveOverseerInput = (input: OverseerVizInput) => {
  const filteredRuns = input.runs.map((run) => filterRunEpisodes(run, input.filters));
  const selectedEpisodeId = pickEpisodeId(filteredRuns, input.selectedEpisodeId, input.preferredRunName);
  const primaryRun = pickPrimaryRun(filteredRuns, input.preferredRunName);
  const primaryEpisode = primaryRun ? getEpisodeForRun(primaryRun, selectedEpisodeId) : null;

  return {
    runs: filteredRuns,
    selectedEpisodeId,
    primaryRun,
    primaryEpisode,
  };
};
