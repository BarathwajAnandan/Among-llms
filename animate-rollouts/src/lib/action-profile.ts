import type {RolloutEvent} from '../types';

type MovePose = 'attack-light' | 'attack-heavy' | 'counter' | 'guard';

type MovePattern = {
  pattern: RegExp;
  label: string;
  lunge: number;
  knockback: number;
  shake: number;
  pose: MovePose;
  color: string;
};

const attackerPatterns: MovePattern[] = [
  {
    pattern: /payload|zero[ _-]?day|exploit|breach|slam|drop|escalate/i,
    label: 'Breach Slam',
    lunge: 170,
    knockback: 200,
    shake: 22,
    pose: 'attack-heavy',
    color: '#ff7b4a',
  },
  {
    pattern: /trap|bait|decoy|inject|spawn|mutate|noise|jam|lure/i,
    label: 'Trap Feint',
    lunge: 130,
    knockback: 130,
    shake: 14,
    pose: 'attack-light',
    color: '#f7b551',
  },
  {
    pattern: /stealth|probe|scan|poke|pressure/i,
    label: 'Probe Jab',
    lunge: 120,
    knockback: 120,
    shake: 11,
    pose: 'attack-light',
    color: '#f6d55d',
  },
];

const defenderPatterns: MovePattern[] = [
  {
    pattern: /contain|lock|patch|sandbox|shield|isolate|quarantine|least[_-]?privilege/i,
    label: 'Hard Block',
    lunge: 118,
    knockback: 110,
    shake: 10,
    pose: 'guard',
    color: '#76efbc',
  },
  {
    pattern: /replan|adaptive|reroute|evade|fallback|recover/i,
    label: 'Footwork Pivot',
    lunge: 150,
    knockback: 145,
    shake: 15,
    pose: 'counter',
    color: '#5fd6ff',
  },
  {
    pattern: /scan|audit|verify|integrity|policy|monitor|check/i,
    label: 'Read Counter',
    lunge: 132,
    knockback: 138,
    shake: 12,
    pose: 'counter',
    color: '#9ce39b',
  },
];

const defaultAttackerPattern: MovePattern = {
  pattern: /.*/,
  label: 'Pressure Rush',
  lunge: 135,
  knockback: 140,
  shake: 14,
  pose: 'attack-light',
  color: '#f5a257',
};

const defaultDefenderPattern: MovePattern = {
  pattern: /.*/,
  label: 'Stability Guard',
  lunge: 120,
  knockback: 120,
  shake: 10,
  pose: 'guard',
  color: '#7fe3cb',
};

const pickPattern = (value: string, patterns: MovePattern[], fallback: MovePattern): MovePattern => {
  const match = patterns.find((entry) => entry.pattern.test(value));
  return match ?? fallback;
};

const toDisplayText = (value: string): string => {
  return value
    .replace(/[_-]/g, ' ')
    .replace(/\s+/g, ' ')
    .trim();
};

export type ExchangeProfile = {
  attackerLabel: string;
  defenderLabel: string;
  attackerLunge: number;
  defenderLunge: number;
  attackerKnockback: number;
  defenderKnockback: number;
  attackerPose: MovePose;
  defenderPose: MovePose;
  shake: number;
  edge: -1 | 0 | 1;
  impactColor: string;
  ticker: string;
  commentary: string;
};

export const getExchangeProfile = (event: RolloutEvent): ExchangeProfile => {
  const attackerMove = pickPattern(event.attackerAction, attackerPatterns, defaultAttackerPattern);
  const defenderMove = pickPattern(event.defenderAction, defenderPatterns, defaultDefenderPattern);

  const edge = event.compromised ? 1 : event.taskSuccess ? -1 : 0;
  const baseShake = Math.max(attackerMove.shake, defenderMove.shake);
  const impactColor =
    edge > 0 ? attackerMove.color : edge < 0 ? defenderMove.color : event.riskScore && event.riskScore > 0.6 ? '#ffbf69' : '#9fb6cf';

  const ticker = `${attackerMove.label} vs ${defenderMove.label}`;

  const riskNote =
    typeof event.riskScore === 'number' && event.riskScore >= 0.7
      ? ` Risk ${(event.riskScore * 100).toFixed(0)}%.`
      : '';

  const commentary =
    edge > 0
      ? `Compromise detected. ${toDisplayText(event.attackerAction)} breaks through.${riskNote}`
      : edge < 0
        ? `Objective secured. ${toDisplayText(event.defenderAction)} stabilizes the round.`
        : `No decisive hit. ${toDisplayText(event.attackerAction)} meets ${toDisplayText(event.defenderAction)}.${riskNote}`;

  return {
    attackerLabel: attackerMove.label,
    defenderLabel: defenderMove.label,
    attackerLunge: attackerMove.lunge,
    defenderLunge: defenderMove.lunge,
    attackerKnockback: attackerMove.knockback,
    defenderKnockback: defenderMove.knockback,
    attackerPose: attackerMove.pose,
    defenderPose: defenderMove.pose,
    shake: baseShake,
    edge,
    impactColor,
    ticker,
    commentary,
  };
};
