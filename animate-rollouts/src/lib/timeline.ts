import type {
  CompiledMatch,
  CompiledRound,
  CompiledStep,
  FrameContext,
  MatchInput,
  MatchTimingConfig,
  RolloutEvent,
} from '../types';

export const defaultTiming: MatchTimingConfig = {
  introFrames: 150,
  roundIntroFrames: 64,
  betweenRoundsFrames: 44,
  finaleFrames: 180,
  step: {
    windupFrames: 26,
    impactFrames: 22,
    recoveryFrames: 30,
    holdFrames: 26,
  },
  emphasisBonusFrames: 20,
};

const mergeTiming = (input?: MatchInput['timing']): MatchTimingConfig => {
  if (!input) {
    return defaultTiming;
  }

  return {
    introFrames: input.introFrames ?? defaultTiming.introFrames,
    roundIntroFrames: input.roundIntroFrames ?? defaultTiming.roundIntroFrames,
    betweenRoundsFrames: input.betweenRoundsFrames ?? defaultTiming.betweenRoundsFrames,
    finaleFrames: input.finaleFrames ?? defaultTiming.finaleFrames,
    emphasisBonusFrames: input.emphasisBonusFrames ?? defaultTiming.emphasisBonusFrames,
    step: {
      windupFrames: input.step?.windupFrames ?? defaultTiming.step.windupFrames,
      impactFrames: input.step?.impactFrames ?? defaultTiming.step.impactFrames,
      recoveryFrames: input.step?.recoveryFrames ?? defaultTiming.step.recoveryFrames,
      holdFrames: input.step?.holdFrames ?? defaultTiming.step.holdFrames,
    },
  };
};

const groupIntoRounds = (events: RolloutEvent[]): RolloutEvent[][] => {
  if (events.length === 0) {
    return [];
  }

  const rounds: RolloutEvent[][] = [];
  let currentRound: RolloutEvent[] = [];

  for (const event of events) {
    const lastEvent = currentRound[currentRound.length - 1];
    if (!lastEvent) {
      currentRound.push(event);
      continue;
    }

    const environmentChanged = event.env !== lastEvent.env;
    const episodeChanged = event.episode !== lastEvent.episode;

    if (environmentChanged || episodeChanged) {
      rounds.push(currentRound);
      currentRound = [event];
      continue;
    }

    currentRound.push(event);
  }

  if (currentRound.length > 0) {
    rounds.push(currentRound);
  }

  return rounds;
};

export const compileMatch = (input: MatchInput): CompiledMatch => {
  const timing = mergeTiming(input.timing);

  const sortedEvents = [...input.events].sort((a, b) => {
    if (a.episode !== b.episode) {
      return a.episode - b.episode;
    }

    return a.step - b.step;
  });

  const eventRounds = groupIntoRounds(sortedEvents);

  let frame = 0;
  const introStartFrame = frame;
  const introEndFrame = introStartFrame + timing.introFrames;
  frame = introEndFrame;

  let attackerScore = 0;
  let defenderScore = 0;

  const rounds: CompiledRound[] = [];
  const transitions: CompiledMatch['transitions'] = [];

  eventRounds.forEach((roundEvents, roundIndex) => {
    const startFrame = frame;
    const introEndForRound = startFrame + timing.roundIntroFrames;
    frame = introEndForRound;

    const steps: CompiledStep[] = [];

    for (const event of roundEvents) {
      const emphasis = event.compromised || event.taskSuccess ? timing.emphasisBonusFrames : 0;
      const durationInFrames =
        timing.step.windupFrames +
        timing.step.impactFrames +
        timing.step.recoveryFrames +
        timing.step.holdFrames +
        emphasis;

      const start = frame;
      const windupEndFrame = start + timing.step.windupFrames;
      const impactEndFrame = windupEndFrame + timing.step.impactFrames;
      const recoveryEndFrame = impactEndFrame + timing.step.recoveryFrames;
      const end = start + durationInFrames;

      const attackerDelta = event.compromised ? 1 : 0;
      const defenderDelta = event.taskSuccess && !event.compromised ? 1 : 0;

      attackerScore += attackerDelta;
      defenderScore += defenderDelta;

      steps.push({
        event,
        startFrame: start,
        endFrame: end,
        durationInFrames,
        windupEndFrame,
        impactEndFrame,
        recoveryEndFrame,
        attackerDelta,
        defenderDelta,
        cumulativeAttackerScore: attackerScore,
        cumulativeDefenderScore: defenderScore,
      });

      frame = end;
    }

    const round: CompiledRound = {
      roundNumber: roundIndex + 1,
      episode: roundEvents[0]?.episode ?? roundIndex + 1,
      env: roundEvents[0]?.env ?? `Environment-${roundIndex + 1}`,
      startFrame,
      introEndFrame: introEndForRound,
      endFrame: frame,
      steps,
      attackerScore,
      defenderScore,
    };

    rounds.push(round);

    const nextRound = eventRounds[roundIndex + 1];
    if (nextRound) {
      transitions.push({
        startFrame: frame,
        endFrame: frame + timing.betweenRoundsFrames,
        fromRoundNumber: round.roundNumber,
        toRoundNumber: round.roundNumber + 1,
        fromEnv: round.env,
        toEnv: nextRound[0]?.env ?? round.env,
      });

      frame += timing.betweenRoundsFrames;
    }
  });

  const finaleStartFrame = frame;
  const finaleEndFrame = finaleStartFrame + timing.finaleFrames;
  const totalFrames = finaleEndFrame;

  const winner =
    attackerScore === defenderScore
      ? 'draw'
      : attackerScore > defenderScore
        ? 'attacker'
        : 'defender';

  return {
    timing,
    introStartFrame,
    introEndFrame,
    rounds,
    transitions,
    finaleStartFrame,
    finaleEndFrame,
    totalFrames,
    attackerScore,
    defenderScore,
    winner,
  };
};

export const getFrameContext = (compiled: CompiledMatch, frame: number): FrameContext => {
  if (frame < compiled.introEndFrame) {
    return {scene: 'intro'};
  }

  if (frame >= compiled.finaleStartFrame) {
    return {
      scene: 'finale',
      finalFrame: frame - compiled.finaleStartFrame,
    };
  }

  for (const transition of compiled.transitions) {
    if (frame >= transition.startFrame && frame < transition.endFrame) {
      return {
        scene: 'transition',
        transition,
      };
    }
  }

  for (const round of compiled.rounds) {
    if (frame < round.startFrame || frame >= round.endFrame) {
      continue;
    }

    const roundFrame = frame - round.startFrame;
    if (frame < round.introEndFrame) {
      return {
        scene: 'round-intro',
        round,
        roundFrame,
      };
    }

    for (let index = 0; index < round.steps.length; index += 1) {
      const step = round.steps[index];
      if (frame >= step.startFrame && frame < step.endFrame) {
        return {
          scene: 'action',
          round,
          step,
          roundFrame,
          stepFrame: frame - step.startFrame,
          stepIndex: index,
        };
      }
    }

    const fallbackStep = round.steps[round.steps.length - 1];
    if (fallbackStep) {
      return {
        scene: 'action',
        round,
        step: fallbackStep,
        roundFrame,
        stepFrame: Math.max(0, frame - fallbackStep.startFrame),
        stepIndex: round.steps.length - 1,
      };
    }
  }

  return {
    scene: 'finale',
    finalFrame: Math.max(0, frame - compiled.finaleStartFrame),
  };
};

export const getScoreAtFrame = (
  compiled: CompiledMatch,
  frame: number,
): {attackerScore: number; defenderScore: number} => {
  let attackerScore = 0;
  let defenderScore = 0;

  for (const round of compiled.rounds) {
    for (const step of round.steps) {
      if (frame < step.endFrame) {
        return {attackerScore, defenderScore};
      }

      attackerScore = step.cumulativeAttackerScore;
      defenderScore = step.cumulativeDefenderScore;
    }
  }

  return {attackerScore, defenderScore};
};
