import type {FighterPose} from '../lib/sprite-data';

type AmongUsActorProps = {
  side: 'attacker' | 'defender';
  x: number;
  baselineY: number;
  pose: FighterPose;
  localFrame: number;
};

const BODY_WIDTH = 120;
const BODY_HEIGHT = 150;
const SCALE = 1.4;

const poseTransforms: Record<FighterPose, (frame: number) => {
  bodyRotate: number;
  bodyOffsetX: number;
  bodyOffsetY: number;
  legSpread: number;
  visorFlash: number;
  squishX: number;
  squishY: number;
}> = {
  idle: (frame) => ({
    bodyRotate: 0,
    bodyOffsetX: 0,
    bodyOffsetY: Math.sin(frame / 8) * 3,
    legSpread: 0,
    visorFlash: 0,
    squishX: 1,
    squishY: 1,
  }),
  walk: (frame) => ({
    bodyRotate: Math.sin(frame / 5) * 6,
    bodyOffsetX: 0,
    bodyOffsetY: Math.abs(Math.sin(frame / 4)) * -6,
    legSpread: Math.sin(frame / 4) * 12,
    visorFlash: 0,
    squishX: 1,
    squishY: 1,
  }),
  guard: (frame) => ({
    bodyRotate: -5,
    bodyOffsetX: -4,
    bodyOffsetY: Math.sin(frame / 10) * 2 + 4,
    legSpread: 6,
    visorFlash: 0,
    squishX: 1.05,
    squishY: 0.96,
  }),
  'attack-light': (frame) => ({
    bodyRotate: 18,
    bodyOffsetX: 14,
    bodyOffsetY: -4,
    legSpread: 8,
    visorFlash: 0.6,
    squishX: 0.92,
    squishY: 1.06,
  }),
  'attack-heavy': (frame) => ({
    bodyRotate: 28,
    bodyOffsetX: 22,
    bodyOffsetY: -8,
    legSpread: 14,
    visorFlash: 1,
    squishX: 0.86,
    squishY: 1.12,
  }),
  counter: (frame) => ({
    bodyRotate: -12,
    bodyOffsetX: -10,
    bodyOffsetY: -6,
    legSpread: 10,
    visorFlash: 0.4,
    squishX: 0.94,
    squishY: 1.04,
  }),
  hurt: (frame) => {
    const fallProgress = Math.min(1, frame / 10);
    const wobble = Math.sin(frame / 2) * 4 * (1 - fallProgress * 0.6);
    return {
      bodyRotate: -70 * fallProgress + wobble,
      bodyOffsetX: -18 * fallProgress,
      bodyOffsetY: 30 * fallProgress,
      legSpread: 18 * fallProgress,
      visorFlash: 0,
      squishX: 1 + 0.15 * fallProgress,
      squishY: 1 - 0.2 * fallProgress,
    };
  },
  recovery: (frame) => {
    const t = Math.min(1, frame / 18);
    const eased = t * t * (3 - 2 * t);
    return {
      bodyRotate: -70 * (1 - eased),
      bodyOffsetX: -18 * (1 - eased),
      bodyOffsetY: 30 * (1 - eased),
      legSpread: 18 * (1 - eased),
      visorFlash: 0,
      squishX: 1 + 0.15 * (1 - eased),
      squishY: 1 - 0.2 * (1 - eased),
    };
  },
  victory: (frame) => ({
    bodyRotate: Math.sin(frame / 4) * 8,
    bodyOffsetX: 0,
    bodyOffsetY: -Math.abs(Math.sin(frame / 5)) * 18,
    legSpread: Math.sin(frame / 4) * 10,
    visorFlash: 0.3 + Math.sin(frame / 3) * 0.3,
    squishX: 1,
    squishY: 1,
  }),
};

const ATTACKER_BODY = '#e84444';
const ATTACKER_BODY_DARK = '#b02e2e';
const DEFENDER_BODY = '#2ed8a0';
const DEFENDER_BODY_DARK = '#1a9e74';

const VISOR_COLOR = '#7cc8e8';
const VISOR_HIGHLIGHT = '#c2e8f8';
const VISOR_FLASH = '#ffffff';

export const AmongUsActor = ({side, x, baselineY, pose, localFrame}: AmongUsActorProps) => {
  const mirror = side === 'attacker' ? 1 : -1;
  const poseFunc = poseTransforms[pose] ?? poseTransforms.idle;
  const p = poseFunc(localFrame);

  const bodyColor = side === 'attacker' ? ATTACKER_BODY : DEFENDER_BODY;
  const bodyDark = side === 'attacker' ? ATTACKER_BODY_DARK : DEFENDER_BODY_DARK;

  const totalWidth = BODY_WIDTH * SCALE;
  const totalHeight = (BODY_HEIGHT + 40) * SCALE;

  const visorGlow = p.visorFlash > 0
    ? `drop-shadow(0 0 ${4 + p.visorFlash * 8}px ${VISOR_HIGHLIGHT})`
    : 'none';

  return (
    <div
      style={{
        position: 'absolute',
        left: x - totalWidth / 2 + p.bodyOffsetX * mirror,
        top: baselineY - totalHeight + p.bodyOffsetY,
        width: totalWidth,
        height: totalHeight,
        transform: `scaleX(${mirror})`,
        transformOrigin: '50% 100%',
      }}
    >
      {/* Shadow */}
      <div
        style={{
          position: 'absolute',
          left: totalWidth * 0.15,
          right: totalWidth * 0.15,
          bottom: -4,
          height: 14,
          borderRadius: 999,
          backgroundColor: 'rgba(0, 0, 0, 0.35)',
          filter: 'blur(5px)',
        }}
      />

      <svg
        width={totalWidth}
        height={totalHeight}
        viewBox="0 0 120 190"
        style={{
          overflow: 'visible',
          filter: visorGlow,
        }}
      >
        <defs>
          <linearGradient id={`bodyGrad-${side}`} x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor={bodyColor} />
            <stop offset="100%" stopColor={bodyDark} />
          </linearGradient>
          <clipPath id={`bodyClip-${side}`}>
            <path d="M20,55 Q20,10 60,10 Q100,10 100,55 L100,120 Q100,145 80,145 L40,145 Q20,145 20,120 Z" />
          </clipPath>
        </defs>

        <g transform={`
          translate(60, 95)
          rotate(${p.bodyRotate})
          scale(${p.squishX}, ${p.squishY})
          translate(-60, -95)
        `}>
          {/* Backpack */}
          <rect
            x="0"
            y="58"
            width="22"
            height="50"
            rx="10"
            ry="10"
            fill={bodyDark}
            stroke={bodyDark}
            strokeWidth="1"
          />

          {/* Main body */}
          <path
            d="M20,55 Q20,10 60,10 Q100,10 100,55 L100,120 Q100,145 80,145 L40,145 Q20,145 20,120 Z"
            fill={`url(#bodyGrad-${side})`}
            stroke={bodyDark}
            strokeWidth="2"
          />

          {/* Visor */}
          <g clipPath={`url(#bodyClip-${side})`}>
            <rect
              x="52"
              y="38"
              width="50"
              height="32"
              rx="12"
              ry="12"
              fill={p.visorFlash > 0
                ? lerpColor(VISOR_COLOR, VISOR_FLASH, p.visorFlash)
                : VISOR_COLOR}
              stroke="rgba(0,0,0,0.15)"
              strokeWidth="1"
            />
            {/* Visor highlight */}
            <rect
              x="75"
              y="42"
              width="14"
              height="6"
              rx="3"
              ry="3"
              fill={VISOR_HIGHLIGHT}
              opacity={0.7}
            />
          </g>

          {/* Left leg */}
          <rect
            x="30"
            y="140"
            width="22"
            height="36"
            rx="8"
            ry="8"
            fill={bodyColor}
            stroke={bodyDark}
            strokeWidth="1.5"
            transform={`translate(${-p.legSpread}, 0)`}
          />
          {/* Left shoe */}
          <ellipse
            cx={41 - p.legSpread}
            cy="176"
            rx="14"
            ry="8"
            fill={bodyDark}
          />

          {/* Right leg */}
          <rect
            x="68"
            y="140"
            width="22"
            height="36"
            rx="8"
            ry="8"
            fill={bodyColor}
            stroke={bodyDark}
            strokeWidth="1.5"
            transform={`translate(${p.legSpread}, 0)`}
          />
          {/* Right shoe */}
          <ellipse
            cx={79 + p.legSpread}
            cy="176"
            rx="14"
            ry="8"
            fill={bodyDark}
          />
        </g>
      </svg>
    </div>
  );
};

function lerpColor(color1: string, color2: string, t: number): string {
  const c1 = hexToRgb(color1);
  const c2 = hexToRgb(color2);
  const r = Math.round(c1.r + (c2.r - c1.r) * t);
  const g = Math.round(c1.g + (c2.g - c1.g) * t);
  const b = Math.round(c1.b + (c2.b - c1.b) * t);
  return `rgb(${r}, ${g}, ${b})`;
}

function hexToRgb(hex: string): {r: number; g: number; b: number} {
  const value = hex.replace('#', '');
  return {
    r: Number.parseInt(value.slice(0, 2), 16),
    g: Number.parseInt(value.slice(2, 4), 16),
    b: Number.parseInt(value.slice(4, 6), 16),
  };
}
