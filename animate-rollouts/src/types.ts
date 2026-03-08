export type RolloutEvent = {
  episode: number;
  step: number;
  env: string;
  attackerAction: string;
  defenderAction: string;
  compromised: boolean;
  taskSuccess: boolean;
  reward?: number;
  riskScore?: number;
  timestamp?: string;
};

export type StepTiming = {
  windupFrames: number;
  impactFrames: number;
  recoveryFrames: number;
  holdFrames: number;
};

export type MatchTimingConfig = {
  introFrames: number;
  roundIntroFrames: number;
  betweenRoundsFrames: number;
  finaleFrames: number;
  step: StepTiming;
  emphasisBonusFrames: number;
};

export type CharacterStyle = 'sprite' | 'among-us';

export type MatchInput = {
  title: string;
  attackerName: string;
  defenderName: string;
  broadcastMode?: boolean;
  characterStyle?: CharacterStyle;
  events: RolloutEvent[];
  timing?: Partial<MatchTimingConfig> & {
    step?: Partial<StepTiming>;
  };
};

export type CompiledStep = {
  event: RolloutEvent;
  startFrame: number;
  endFrame: number;
  durationInFrames: number;
  windupEndFrame: number;
  impactEndFrame: number;
  recoveryEndFrame: number;
  attackerDelta: number;
  defenderDelta: number;
  cumulativeAttackerScore: number;
  cumulativeDefenderScore: number;
};

export type CompiledRound = {
  roundNumber: number;
  episode: number;
  env: string;
  startFrame: number;
  introEndFrame: number;
  endFrame: number;
  steps: CompiledStep[];
  attackerScore: number;
  defenderScore: number;
};

export type RoundTransition = {
  startFrame: number;
  endFrame: number;
  fromRoundNumber: number;
  toRoundNumber: number;
  fromEnv: string;
  toEnv: string;
};

export type CompiledMatch = {
  timing: MatchTimingConfig;
  introStartFrame: number;
  introEndFrame: number;
  rounds: CompiledRound[];
  transitions: RoundTransition[];
  finaleStartFrame: number;
  finaleEndFrame: number;
  totalFrames: number;
  attackerScore: number;
  defenderScore: number;
  winner: 'attacker' | 'defender' | 'draw';
};

export type FrameContext =
  | {scene: 'intro'}
  | {
      scene: 'round-intro';
      round: CompiledRound;
      roundFrame: number;
    }
  | {
      scene: 'action';
      round: CompiledRound;
      step: CompiledStep;
      roundFrame: number;
      stepFrame: number;
      stepIndex: number;
    }
  | {
      scene: 'transition';
      transition: RoundTransition;
    }
  | {
      scene: 'finale';
      finalFrame: number;
    };
