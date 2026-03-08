export type FighterPose =
  | 'idle'
  | 'walk'
  | 'guard'
  | 'attack-light'
  | 'attack-heavy'
  | 'counter'
  | 'hurt'
  | 'recovery'
  | 'victory';

export type FighterSheet = {
  src: string;
  frameWidth: number;
  frameHeight: number;
  columns: number;
  rows: number;
  scale: number;
  animations: Record<FighterPose, number[]>;
};

const catAnimationFrames: Record<FighterPose, number[]> = {
  idle: [0, 1, 2, 3],
  walk: [10, 11, 12, 13, 14, 15],
  guard: [20, 21, 22, 23],
  'attack-light': [24, 25, 26, 27],
  'attack-heavy': [30, 31, 32, 33],
  counter: [34, 35, 36, 37],
  hurt: [40, 41, 42],
  recovery: [43, 44, 45],
  victory: [50, 51, 52, 53],
};

export const fighterSheets: Record<'attacker' | 'defender', FighterSheet> = {
  attacker: {
    src: 'assets/sprites/cat_fighter_sprite1.png',
    frameWidth: 50,
    frameHeight: 50,
    columns: 10,
    rows: 10,
    scale: 4.2,
    animations: catAnimationFrames,
  },
  defender: {
    src: 'assets/sprites/cat_fighter_sprite2.png',
    frameWidth: 50,
    frameHeight: 50,
    columns: 10,
    rows: 10,
    scale: 3.9,
    animations: catAnimationFrames,
  },
};

export const pickFrame = (
  frames: number[],
  localFrame: number,
  framesPerSprite: number,
): number => {
  if (frames.length === 0) {
    return 0;
  }

  const safeFrameSpan = Math.max(1, Math.floor(framesPerSprite));
  const index = Math.floor(localFrame / safeFrameSpan) % frames.length;
  return frames[index] ?? frames[0] ?? 0;
};
