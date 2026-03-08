import {Img, staticFile} from 'remotion';
import {fighterSheets, pickFrame, type FighterPose} from '../lib/sprite-data';

const poseFramesPerSprite: Record<FighterPose, number> = {
  idle: 12,
  walk: 8,
  guard: 9,
  'attack-light': 7,
  'attack-heavy': 6,
  counter: 7,
  hurt: 10,
  recovery: 10,
  victory: 8,
};

const hexToRgba = (hex: string, alpha: number): string => {
  const value = hex.replace('#', '');
  if (value.length !== 6) {
    return `rgba(255, 255, 255, ${alpha})`;
  }
  const r = Number.parseInt(value.slice(0, 2), 16);
  const g = Number.parseInt(value.slice(2, 4), 16);
  const b = Number.parseInt(value.slice(4, 6), 16);
  return `rgba(${r}, ${g}, ${b}, ${alpha})`;
};

type SpriteActorProps = {
  side: 'attacker' | 'defender';
  x: number;
  baselineY: number;
  pose: FighterPose;
  localFrame: number;
};

export const SpriteActor = ({side, x, baselineY, pose, localFrame}: SpriteActorProps) => {
  const sheet = fighterSheets[side];
  const animation = sheet.animations[pose] ?? sheet.animations.idle;
  const frameIndex = pickFrame(animation, localFrame, poseFramesPerSprite[pose]);
  const frameColumn = frameIndex % sheet.columns;
  const frameRow = Math.floor(frameIndex / sheet.columns);
  const accentColor = side === 'attacker' ? '#ff7382' : '#74ffd2';
  const tintFilter =
    side === 'attacker'
      ? 'hue-rotate(-26deg) saturate(1.55) brightness(1.03) contrast(1.04)'
      : 'hue-rotate(138deg) saturate(1.45) brightness(1.08) contrast(1.03)';

  const actorWidth = sheet.frameWidth * sheet.scale;
  const actorHeight = sheet.frameHeight * sheet.scale;
  const mirror = side === 'attacker' ? 1 : -1;

  return (
    <div
      style={{
        position: 'absolute',
        left: x - actorWidth / 2,
        top: baselineY - actorHeight,
        width: actorWidth,
        height: actorHeight,
      }}
    >
      <div
        style={{
          position: 'absolute',
          left: actorWidth * 0.16,
          right: actorWidth * 0.16,
          bottom: -14,
          height: 20,
          borderRadius: 12,
          backgroundColor: 'rgba(5, 8, 12, 0.44)',
          filter: 'blur(6px)',
        }}
      />
      <div
        style={{
          position: 'absolute',
          left: actorWidth * 0.19,
          right: actorWidth * 0.19,
          bottom: -7,
          height: 12,
          borderRadius: 999,
          border: `2px solid ${accentColor}`,
          backgroundColor: hexToRgba(accentColor, 0.18),
        }}
      />
      <div
        style={{
          position: 'absolute',
          inset: 0,
          overflow: 'hidden',
          transform: `scaleX(${mirror})`,
          transformOrigin: '50% 100%',
          filter: tintFilter,
        }}
      >
        <Img
          src={staticFile(sheet.src)}
          style={{
            position: 'absolute',
            imageRendering: 'pixelated',
            width: sheet.columns * sheet.frameWidth * sheet.scale,
            height: sheet.rows * sheet.frameHeight * sheet.scale,
            left: -frameColumn * sheet.frameWidth * sheet.scale,
            top: -frameRow * sheet.frameHeight * sheet.scale,
          }}
        />
      </div>
    </div>
  );
};
