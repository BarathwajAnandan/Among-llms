import {Audio} from '@remotion/media';
import {useMemo} from 'react';
import {
  AbsoluteFill,
  Easing,
  Img,
  Sequence,
  interpolate,
  spring,
  staticFile,
  useCurrentFrame,
  useVideoConfig,
} from 'remotion';
import {AmongUsActor} from '../components/AmongUsActor';
import {SpriteActor} from '../components/SpriteActor';
import {getExchangeProfile} from '../lib/action-profile';
import {getEnvironmentTheme} from '../lib/env-theme';
import {type FighterPose} from '../lib/sprite-data';
import {compileMatch, getFrameContext, getScoreAtFrame} from '../lib/timeline';
import type {CharacterStyle, CompiledRound, CompiledStep, MatchInput} from '../types';

const clampConfig = {
  extrapolateLeft: 'clamp' as const,
  extrapolateRight: 'clamp' as const,
};

const ringFont = 'Trebuchet MS, Tahoma, Verdana, sans-serif';
const displayFont = 'Impact, Haettenschweiler, Arial Narrow Bold, sans-serif';

const cleanAction = (value: string): string => {
  return value
    .replace(/[_-]/g, ' ')
    .replace(/\s+/g, ' ')
    .trim();
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

const toFighterPose = (pose: 'attack-light' | 'attack-heavy' | 'counter' | 'guard'): FighterPose => {
  if (pose === 'attack-heavy') {
    return 'attack-heavy';
  }

  if (pose === 'attack-light') {
    return 'attack-light';
  }

  if (pose === 'counter') {
    return 'counter';
  }

  return 'guard';
};

type FightState = {
  attackerX: number;
  defenderX: number;
  attackerPose: FighterPose;
  defenderPose: FighterPose;
  cameraShake: number;
  impactOpacity: number;
  impactScale: number;
  impactColor: string;
  ticker: string;
  commentary: string;
};

const POSITIONS: Record<CharacterStyle, {baseAttackerX: number; baseDefenderX: number; contactGap: number}> = {
  'among-us': {baseAttackerX: 830, baseDefenderX: 1090, contactGap: 60},
  sprite: {baseAttackerX: 760, baseDefenderX: 1160, contactGap: 95},
};

const getActionState = (step: CompiledStep, stepFrame: number, characterStyle: CharacterStyle = 'among-us'): FightState => {
  const windupFrames = Math.max(1, step.windupEndFrame - step.startFrame);
  const impactFrames = Math.max(1, step.impactEndFrame - step.windupEndFrame);
  const recoveryFrames = Math.max(1, step.recoveryEndFrame - step.impactEndFrame);

  const windupProgress = interpolate(stepFrame, [0, windupFrames], [0, 1], clampConfig);
  const impactProgress = interpolate(
    stepFrame,
    [windupFrames, windupFrames + impactFrames],
    [0, 1],
    clampConfig,
  );
  const recoveryProgress = interpolate(
    stepFrame,
    [windupFrames + impactFrames, windupFrames + impactFrames + recoveryFrames],
    [0, 1],
    clampConfig,
  );

  const profile = getExchangeProfile(step.event);

  const pos = POSITIONS[characterStyle];
  const baseAttackerX = pos.baseAttackerX;
  const baseDefenderX = pos.baseDefenderX;
  const contactCenterX = 960;
  const contactGap = pos.contactGap;
  const attackerContactX = contactCenterX - contactGap;
  const defenderContactX = contactCenterX + contactGap;

  let attackerX = baseAttackerX;
  let defenderX = baseDefenderX;
  let attackerPose: FighterPose = 'guard';
  let defenderPose: FighterPose = 'guard';

  if (profile.edge > 0) {
    const attackerApproach = Math.min(1, windupProgress * 0.86 + impactProgress * 0.58);
    const defenderApproach = Math.min(1, windupProgress * 0.58 + impactProgress * 0.34);
    const defenderKnockback =
      profile.attackerKnockback * 0.38 * impactProgress * (1 - 0.68 * recoveryProgress);

    attackerX =
      baseAttackerX +
      (attackerContactX - baseAttackerX) * attackerApproach -
      26 * recoveryProgress;
    defenderX =
      baseDefenderX +
      (defenderContactX - baseDefenderX) * defenderApproach +
      defenderKnockback;

    attackerPose = impactProgress > 0.1 ? toFighterPose(profile.attackerPose) : 'walk';
    defenderPose = impactProgress > 0.08 ? 'hurt' : 'guard';

    if (recoveryProgress > 0.62) {
      attackerPose = 'recovery';
      defenderPose = 'recovery';
    }
  } else if (profile.edge < 0) {
    const defenderApproach = Math.min(1, windupProgress * 0.86 + impactProgress * 0.58);
    const attackerApproach = Math.min(1, windupProgress * 0.58 + impactProgress * 0.34);
    const attackerKnockback =
      profile.defenderKnockback * 0.38 * impactProgress * (1 - 0.68 * recoveryProgress);

    attackerX =
      baseAttackerX +
      (attackerContactX - baseAttackerX) * attackerApproach -
      attackerKnockback;
    defenderX =
      baseDefenderX +
      (defenderContactX - baseDefenderX) * defenderApproach +
      26 * recoveryProgress;

    attackerPose = impactProgress > 0.08 ? 'hurt' : 'guard';
    defenderPose = impactProgress > 0.1 ? toFighterPose(profile.defenderPose) : 'walk';

    if (recoveryProgress > 0.62) {
      attackerPose = 'recovery';
      defenderPose = 'recovery';
    }
  } else {
    const drift = Math.sin(stepFrame / 11) * 8;
    const converge = 22 * windupProgress;
    attackerX = baseAttackerX + drift + converge;
    defenderX = baseDefenderX - drift - converge;
    attackerPose = 'guard';
    defenderPose = 'guard';
  }

  const cameraShake =
    Math.sin(stepFrame * 1.9) *
    profile.shake *
    impactProgress *
    (1 - 0.72 * recoveryProgress) *
    0.33;

  return {
    attackerX,
    defenderX,
    attackerPose,
    defenderPose,
    cameraShake,
    impactOpacity: impactProgress * (profile.edge === 0 ? 0.24 : 0.56),
    impactScale: 0.44 + impactProgress * 0.72,
    impactColor: profile.impactColor,
    ticker: profile.ticker,
    commentary: profile.commentary,
  };
};

const IntroOverlay = ({
  frame,
  title,
  attackerName,
  defenderName,
}: {
  frame: number;
  title: string;
  attackerName: string;
  defenderName: string;
}) => {
  const {fps} = useVideoConfig();
  const introReveal = spring({
    frame,
    fps,
    config: {damping: 200},
    durationInFrames: 28,
  });

  const panelOpacity = interpolate(frame, [0, 18], [0, 1], clampConfig);

  return (
    <AbsoluteFill style={{justifyContent: 'center', alignItems: 'center', gap: 24}}>
      <div
        style={{
          opacity: panelOpacity,
          transform: `scale(${0.9 + 0.1 * introReveal})`,
          backgroundColor: 'rgba(6, 10, 18, 0.86)',
          border: '3px solid #f1c962',
          borderRadius: 18,
          padding: '22px 36px',
          textAlign: 'center',
          color: '#f6fbff',
          fontFamily: displayFont,
          letterSpacing: 1,
          fontSize: 56,
          maxWidth: 1220,
        }}
      >
        {title}
      </div>
      <div style={{display: 'flex', alignItems: 'center', gap: 18, color: '#f1f8ff'}}>
        <div
          style={{
            minWidth: 380,
            textAlign: 'center',
            fontFamily: ringFont,
            fontSize: 32,
            fontWeight: 700,
            borderRadius: 12,
            border: '2px solid #ff8a8a',
            backgroundColor: 'rgba(58, 18, 22, 0.8)',
            padding: '12px 16px',
          }}
        >
          {attackerName}
        </div>
        <div style={{fontFamily: displayFont, fontSize: 44, color: '#f8dd62'}}>VS</div>
        <div
          style={{
            minWidth: 380,
            textAlign: 'center',
            fontFamily: ringFont,
            fontSize: 32,
            fontWeight: 700,
            borderRadius: 12,
            border: '2px solid #84ffc9',
            backgroundColor: 'rgba(14, 44, 34, 0.8)',
            padding: '12px 16px',
          }}
        >
          {defenderName}
        </div>
      </div>
    </AbsoluteFill>
  );
};

const RoundOverlay = ({roundNumber, envName}: {roundNumber: number; envName: string}) => {
  return (
    <AbsoluteFill style={{justifyContent: 'center', alignItems: 'center'}}>
      <div
        style={{
          backgroundColor: 'rgba(6, 11, 21, 0.88)',
          border: '3px solid #f8dd62',
          borderRadius: 16,
          padding: '18px 34px',
          textAlign: 'center',
          color: '#f8fbff',
        }}
      >
        <div style={{fontFamily: displayFont, fontSize: 46, letterSpacing: 2}}>ROUND {roundNumber}</div>
        <div style={{fontFamily: ringFont, fontSize: 26, marginTop: 8}}>{envName}</div>
      </div>
    </AbsoluteFill>
  );
};

const TransitionOverlay = ({fromEnv, toEnv}: {fromEnv: string; toEnv: string}) => {
  return (
    <AbsoluteFill style={{justifyContent: 'center', alignItems: 'center'}}>
      <div
        style={{
          backgroundColor: 'rgba(3, 8, 13, 0.84)',
          border: '2px solid #6fd7ff',
          borderRadius: 14,
          padding: '16px 26px',
          textAlign: 'center',
          color: '#dff7ff',
          minWidth: 780,
        }}
      >
        <div style={{fontFamily: displayFont, fontSize: 36, letterSpacing: 1}}>ENVIRONMENT SHIFT</div>
        <div style={{fontFamily: ringFont, fontSize: 24, marginTop: 10}}>
          {fromEnv} {'->'} {toEnv}
        </div>
      </div>
    </AbsoluteFill>
  );
};

const FinalOverlay = ({
  frame,
  winner,
  attackerName,
  defenderName,
  attackerScore,
  defenderScore,
}: {
  frame: number;
  winner: 'attacker' | 'defender' | 'draw';
  attackerName: string;
  defenderName: string;
  attackerScore: number;
  defenderScore: number;
}) => {
  const reveal = interpolate(frame, [0, 24], [0, 1], {
    ...clampConfig,
    easing: Easing.out(Easing.quad),
  });

  const title =
    winner === 'draw'
      ? 'MATCH DRAW'
      : winner === 'attacker'
        ? `${attackerName} WINS`
        : `${defenderName} WINS`;

  return (
    <AbsoluteFill style={{justifyContent: 'center', alignItems: 'center'}}>
      <div
        style={{
          transform: `scale(${0.94 + 0.06 * reveal})`,
          opacity: reveal,
          backgroundColor: 'rgba(7, 12, 24, 0.88)',
          border: '3px solid #f8dd62',
          borderRadius: 22,
          padding: '26px 40px',
          textAlign: 'center',
          color: '#f9fcff',
          minWidth: 760,
        }}
      >
        <div style={{fontFamily: displayFont, fontSize: 50, letterSpacing: 2}}>{title}</div>
        <div style={{fontFamily: ringFont, fontSize: 26, marginTop: 14}}>
          Compromises: {attackerScore} | Successful Tasks: {defenderScore}
        </div>
      </div>
    </AbsoluteFill>
  );
};

const BroadcastMini = ({
  round,
  frame,
  totalFrames,
  risk,
  ticker,
}: {
  round: CompiledRound | null;
  frame: number;
  totalFrames: number;
  risk: number;
  ticker: string;
}) => {
  const progress = totalFrames > 1 ? Math.max(0, Math.min(1, frame / (totalFrames - 1))) : 0;
  const riskSafe = Math.max(0, Math.min(1, risk));

  return (
    <div
      style={{
        position: 'absolute',
        right: 22,
        top: 132,
        width: 280,
        borderRadius: 12,
        backgroundColor: 'rgba(7, 13, 24, 0.86)',
        border: '1px solid rgba(126, 176, 226, 0.56)',
        padding: '10px 11px',
        color: '#e7f4ff',
      }}
    >
      <div style={{fontFamily: displayFont, fontSize: 24, letterSpacing: 1}}>BROADCAST</div>
      <div style={{fontFamily: ringFont, fontSize: 14, opacity: 0.84, marginTop: 2}}>
        {round ? `Round ${round.roundNumber} - ${round.env}` : 'No active round'}
      </div>
      <div style={{marginTop: 8, fontSize: 13}}>Timeline</div>
      <div
        style={{
          marginTop: 4,
          height: 8,
          borderRadius: 999,
          backgroundColor: 'rgba(23, 47, 72, 0.9)',
          overflow: 'hidden',
        }}
      >
        <div
          style={{
            width: `${(progress * 100).toFixed(2)}%`,
            height: '100%',
            backgroundColor: '#8ed2ff',
          }}
        />
      </div>
      <div style={{marginTop: 8, fontSize: 13}}>Risk {(riskSafe * 100).toFixed(0)}%</div>
      <div
        style={{
          marginTop: 4,
          height: 8,
          borderRadius: 999,
          backgroundColor: 'rgba(37, 45, 57, 0.9)',
          overflow: 'hidden',
        }}
      >
        <div
          style={{
            width: `${(riskSafe * 100).toFixed(2)}%`,
            height: '100%',
            backgroundColor: riskSafe >= 0.7 ? '#ff8f84' : '#ffd276',
          }}
        />
      </div>
      <div style={{marginTop: 8, fontSize: 12, opacity: 0.86}}>{ticker}</div>
    </div>
  );
};

export const RolloutWrestlingComposition = (props: MatchInput) => {
  const frame = useCurrentFrame();
  const {fps} = useVideoConfig();

  const characterStyle: CharacterStyle = props.characterStyle ?? 'sprite';
  const compiled = useMemo(() => compileMatch(props), [props]);
  const context = getFrameContext(compiled, frame);
  const score = getScoreAtFrame(compiled, frame);

  const currentRound: CompiledRound | null =
    context.scene === 'action' || context.scene === 'round-intro'
      ? context.round
      : context.scene === 'transition'
        ? compiled.rounds.find((round) => round.roundNumber === context.transition.toRoundNumber) ?? null
        : null;

  const activeTheme = getEnvironmentTheme(currentRound?.env ?? compiled.rounds[0]?.env ?? 'Arena Prime');

  const defaultPos = POSITIONS[characterStyle];
  let attackerX = defaultPos.baseAttackerX;
  let defenderX = defaultPos.baseDefenderX;
  let attackerPose: FighterPose = 'idle';
  let defenderPose: FighterPose = 'idle';
  let localFrame = frame;
  let cameraShake = 0;
  let impactOpacity = 0;
  let impactScale = 0.5;
  let impactColor = '#ffd36e';
  let impactX = 960;
  let impactY = 472;
  let headline = 'Arena systems online';
  let detail = 'Preparing first exchange.';
  let riskForBroadcast = 0;

  if (context.scene === 'action') {
    const state = getActionState(context.step, context.stepFrame, characterStyle);
    attackerX = state.attackerX;
    defenderX = state.defenderX;
    attackerPose = state.attackerPose;
    defenderPose = state.defenderPose;
    localFrame = context.stepFrame;
    cameraShake = state.cameraShake;
    impactOpacity = state.impactOpacity;
    impactScale = state.impactScale;
    impactColor = state.impactColor;
    impactX = (state.attackerX + state.defenderX) / 2;
    impactY = 468;
    headline = state.ticker;
    detail = state.commentary;
    riskForBroadcast = context.step.event.riskScore ?? 0;
  }

  if (context.scene === 'round-intro') {
    const sway = Math.sin((frame + context.round.roundNumber * 11) / 9) * 16;
    attackerX += sway;
    defenderX -= sway;
    attackerPose = 'walk';
    defenderPose = 'walk';
    localFrame = context.roundFrame;
    headline = `Round ${context.round.roundNumber} starts in ${context.round.env}`;
    detail = `Episode ${context.round.episode} enters the ring.`;
  }

  if (context.scene === 'transition') {
    attackerPose = 'walk';
    defenderPose = 'walk';
    headline = `Transition: ${context.transition.fromEnv} -> ${context.transition.toEnv}`;
    detail = 'Arena rotates for the next rollout environment.';
  }

  if (context.scene === 'finale') {
    attackerPose = compiled.winner === 'attacker' ? 'victory' : compiled.winner === 'defender' ? 'hurt' : 'guard';
    defenderPose = compiled.winner === 'defender' ? 'victory' : compiled.winner === 'attacker' ? 'hurt' : 'guard';
    localFrame = context.finalFrame;
    headline = 'Final verdict';
    detail = 'Judges close the match with total compromise and task counts.';
  }

  const topCenterText =
    context.scene === 'action' || context.scene === 'round-intro'
      ? `ROUND ${context.round.roundNumber}`
      : context.scene === 'transition'
        ? `ROUND ${context.transition.fromRoundNumber} -> ${context.transition.toRoundNumber}`
        : 'SHOWDOWN';

  const topSubText =
    context.scene === 'action'
      ? `${cleanAction(context.step.event.attackerAction)} | ${cleanAction(context.step.event.defenderAction)}`
      : context.scene === 'round-intro'
        ? context.round.env
        : context.scene === 'transition'
          ? context.transition.toEnv
          : 'Final scene';

  const subtlePulse = 0.5 + 0.5 * Math.sin(frame / 18);

  type SfxCue = {
    key: string;
    from: number;
    src: string;
    volume: number;
    playbackRate?: number;
    durationInFrames: number;
  };

  const sfxCues = useMemo((): SfxCue[] => {
    const cues: SfxCue[] = [];

    for (const round of compiled.rounds) {
      cues.push({
        key: `round-bell-${round.roundNumber}`,
        from: round.startFrame,
        src: 'assets/audio/sfx/round-bell.wav',
        volume: 0.3,
        durationInFrames: 24,
      });

      round.steps.forEach((step, index) => {
        const decisive = step.event.compromised || step.event.taskSuccess;
        if (!decisive) {
          return;
        }

        const impactFrame = step.windupEndFrame;

        cues.push({
          key: `impact-${round.roundNumber}-${step.event.step}-${index}`,
          from: impactFrame,
          src: 'assets/audio/sfx/punch.wav',
          volume: step.event.compromised ? 0.56 : 0.44,
          playbackRate: step.event.compromised ? 0.95 : 1.06,
          durationInFrames: 24,
        });

        if (step.event.taskSuccess && !step.event.compromised) {
          cues.push({
            key: `secure-bell-${round.roundNumber}-${step.event.step}-${index}`,
            from: impactFrame + 5,
            src: 'assets/audio/sfx/round-bell.wav',
            volume: 0.17,
            playbackRate: 1.2,
            durationInFrames: 20,
          });
        }
      });
    }

    return cues;
  }, [compiled]);

  return (
    <AbsoluteFill
      style={{
        fontFamily: ringFont,
        color: activeTheme.textColor,
        backgroundColor: '#05070e',
      }}
    >
      <Audio src={staticFile('assets/audio/music/8BitBattleLoop.ogg')} loop volume={0.14} />
      {sfxCues.map((cue) => (
        <Sequence key={cue.key} from={cue.from} durationInFrames={cue.durationInFrames} premountFor={fps}>
          <Audio
            src={staticFile(cue.src)}
            volume={cue.volume}
            playbackRate={cue.playbackRate ?? 1}
          />
        </Sequence>
      ))}

      <AbsoluteFill
        style={{
          background: `linear-gradient(180deg, ${activeTheme.gradientStart} 0%, ${activeTheme.gradientEnd} 100%)`,
        }}
      />
      <Img
        src={staticFile(activeTheme.backdropAsset)}
        style={{
          position: 'absolute',
          inset: 0,
          width: '100%',
          height: '100%',
          objectFit: 'cover',
          opacity: 0.52,
        }}
      />
      <div
        style={{
          position: 'absolute',
          inset: 0,
          background: `radial-gradient(circle at 50% 30%, ${hexToRgba(
            activeTheme.accentColor,
            0.2 + subtlePulse * 0.07,
          )}, rgba(5,7,13,0.72) 70%)`,
        }}
      />

      <Img
        src={staticFile('assets/ui/crowd-strip.svg')}
        style={{
          position: 'absolute',
          left: 0,
          bottom: 272,
          width: '100%',
          height: 206,
          opacity: 0.44,
        }}
      />

      <div
        style={{
          position: 'absolute',
          left: 152,
          right: 152,
          bottom: 102,
          height: 286,
          borderRadius: 32,
          border: `4px solid ${activeTheme.accentColor}`,
          backgroundColor: activeTheme.floorColor,
          boxShadow: `0 0 34px ${hexToRgba(activeTheme.accentColor, 0.2)}`,
          overflow: 'hidden',
        }}
      >
        <div
          style={{
            position: 'absolute',
            left: 0,
            right: 0,
            top: 36,
            borderTop: `5px solid ${hexToRgba(activeTheme.accentColor, 0.68)}`,
          }}
        />
        <div
          style={{
            position: 'absolute',
            left: 0,
            right: 0,
            top: 80,
            borderTop: `4px solid ${hexToRgba(activeTheme.accentColor, 0.52)}`,
          }}
        />
      </div>

      <div
        style={{
          position: 'absolute',
          inset: 0,
          transform: `translateX(${cameraShake}px)`,
        }}
      >
        {characterStyle === 'among-us' ? (
          <>
            <AmongUsActor side="attacker" x={attackerX} baselineY={790} pose={attackerPose} localFrame={localFrame} />
            <AmongUsActor side="defender" x={defenderX} baselineY={790} pose={defenderPose} localFrame={localFrame} />
          </>
        ) : (
          <>
            <SpriteActor side="attacker" x={attackerX} baselineY={790} pose={attackerPose} localFrame={localFrame} />
            <SpriteActor side="defender" x={defenderX} baselineY={790} pose={defenderPose} localFrame={localFrame} />
          </>
        )}
      </div>

      <div
        style={{
          position: 'absolute',
          left: impactX - 105,
          top: impactY,
          width: 210,
          height: 210,
          borderRadius: 999,
          opacity: impactOpacity,
          transform: `scale(${impactScale})`,
          background: `radial-gradient(circle, ${hexToRgba(impactColor, 0.78)} 0%, ${hexToRgba(
            impactColor,
            0.2,
          )} 52%, rgba(255,255,255,0) 72%)`,
          filter: 'blur(1px)',
        }}
      />
      <Img
        src={staticFile('assets/ui/impact-burst.svg')}
        style={{
          position: 'absolute',
          left: impactX - 105,
          top: impactY,
          width: 210,
          height: 210,
          opacity: impactOpacity,
          transform: `scale(${impactScale})`,
        }}
      />

      <div
        style={{
          position: 'absolute',
          left: 22,
          right: 22,
          top: 18,
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          gap: 14,
        }}
      >
        <div
          style={{
            minWidth: 332,
            borderRadius: 11,
            border: '2px solid #ff8088',
            backgroundColor: 'rgba(42, 10, 15, 0.86)',
            padding: '10px 14px',
          }}
        >
          <div style={{fontSize: 17, fontWeight: 700}}>{props.attackerName}</div>
          <div style={{fontSize: 26, fontFamily: displayFont}}>Compromises: {score.attackerScore}</div>
        </div>

        <div
          style={{
            minWidth: 500,
            borderRadius: 11,
            border: `2px solid ${activeTheme.accentColor}`,
            backgroundColor: 'rgba(7, 14, 28, 0.86)',
            padding: '10px 14px',
            textAlign: 'center',
          }}
        >
          <div style={{fontSize: 18, letterSpacing: 1}}>{props.title}</div>
          <div style={{fontFamily: displayFont, fontSize: 30}}>{topCenterText}</div>
          <div style={{fontSize: 15, opacity: 0.86}}>{topSubText}</div>
        </div>

        <div
          style={{
            minWidth: 332,
            borderRadius: 11,
            border: '2px solid #7fffc7',
            backgroundColor: 'rgba(11, 40, 31, 0.86)',
            padding: '10px 14px',
            textAlign: 'right',
          }}
        >
          <div style={{fontSize: 17, fontWeight: 700}}>{props.defenderName}</div>
          <div style={{fontSize: 26, fontFamily: displayFont}}>Task Wins: {score.defenderScore}</div>
        </div>
      </div>

      <div
        style={{
          position: 'absolute',
          left: 22,
          right: 22,
          bottom: 22,
          borderRadius: 12,
          border: `2px solid ${hexToRgba(activeTheme.accentColor, 0.84)}`,
          backgroundColor: 'rgba(6, 9, 18, 0.84)',
          padding: '12px 16px',
          color: '#e8f6ff',
        }}
      >
        <div style={{fontSize: 21, fontWeight: 700}}>{headline}</div>
        <div style={{marginTop: 4, fontSize: 17, opacity: 0.88}}>{detail}</div>
      </div>

      {props.broadcastMode ? (
        <BroadcastMini
          round={currentRound}
          frame={frame}
          totalFrames={compiled.totalFrames}
          risk={riskForBroadcast}
          ticker={headline}
        />
      ) : null}

      {context.scene === 'intro' ? (
        <IntroOverlay
          frame={frame}
          title={props.title}
          attackerName={props.attackerName}
          defenderName={props.defenderName}
        />
      ) : null}

      {context.scene === 'round-intro' ? (
        <RoundOverlay roundNumber={context.round.roundNumber} envName={context.round.env} />
      ) : null}

      {context.scene === 'transition' ? (
        <TransitionOverlay fromEnv={context.transition.fromEnv} toEnv={context.transition.toEnv} />
      ) : null}

      {context.scene === 'finale' ? (
        <FinalOverlay
          frame={context.finalFrame}
          winner={compiled.winner}
          attackerName={props.attackerName}
          defenderName={props.defenderName}
          attackerScore={compiled.attackerScore}
          defenderScore={compiled.defenderScore}
        />
      ) : null}

      <div
        style={{
          position: 'absolute',
          right: 24,
          bottom: 92,
          fontFamily: displayFont,
          fontSize: 30,
          color: hexToRgba(activeTheme.accentColor, 0.9),
          letterSpacing: 1,
        }}
      >
        {(frame / fps).toFixed(1)}s
      </div>
    </AbsoluteFill>
  );
};
