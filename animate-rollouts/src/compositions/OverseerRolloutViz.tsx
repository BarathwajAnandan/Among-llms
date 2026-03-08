import {Audio} from '@remotion/media';
import type {ReactNode} from 'react';
import {AbsoluteFill, Sequence, interpolate, staticFile, useCurrentFrame} from 'remotion';
import {getEpisodeForRun, resolveOverseerInput} from '../overseer/selectors';
import type {EpisodeResult, OverseerVizInput, RunSummary} from '../overseer/types';

const clampConfig = {
  extrapolateLeft: 'clamp' as const,
  extrapolateRight: 'clamp' as const,
};

const sceneFrames = {
  intro: 120,
  lanes: 240,
  scoreboard: 220,
  compare: 220,
  outro: 90,
};

const metricOrder = [
  'attack_detection',
  'failure_detection',
  'goal_degradation_estimate',
  'risk_level',
  'violation_types',
  'culprit_localization',
  'root_cause',
  'recommended_action',
  'false_alarm_penalty',
];

const formatLabel = (value: string): string => {
  return value
    .replace(/_/g, ' ')
    .replace(/\b\w/g, (char) => char.toUpperCase());
};

const formatPercent = (value?: number): string => {
  if (value === undefined || !Number.isFinite(value)) {
    return '-';
  }

  return `${(value * 100).toFixed(1)}%`;
};

const formatScore = (value?: number): string => {
  if (value === undefined || !Number.isFinite(value)) {
    return '-';
  }

  return value.toFixed(2);
};

const statusColor = (value: boolean | undefined): string => {
  if (value === true) {
    return '#7dffb3';
  }

  if (value === false) {
    return '#ff8c8c';
  }

  return '#ffd482';
};

const chipText = (value: boolean | undefined): string => {
  if (value === true) {
    return 'Correct';
  }

  if (value === false) {
    return 'Incorrect';
  }

  return 'N/A';
};

const getOverseerSceneRanges = () => {
  const introStart = 0;
  const introEnd = introStart + sceneFrames.intro;

  const lanesStart = introEnd;
  const lanesEnd = lanesStart + sceneFrames.lanes;

  const scoreStart = lanesEnd;
  const scoreEnd = scoreStart + sceneFrames.scoreboard;

  const compareStart = scoreEnd;
  const compareEnd = compareStart + sceneFrames.compare;

  const outroStart = compareEnd;
  const outroEnd = outroStart + sceneFrames.outro;

  return {
    introStart,
    introEnd,
    lanesStart,
    lanesEnd,
    scoreStart,
    scoreEnd,
    compareStart,
    compareEnd,
    outroStart,
    outroEnd,
    total: outroEnd,
  };
};

export const getOverseerDurationInFrames = (): number => {
  return getOverseerSceneRanges().total;
};

const Panel = ({children}: {children: ReactNode}) => {
  return (
    <div
      style={{
        borderRadius: 18,
        border: '2px solid rgba(165, 202, 236, 0.55)',
        backgroundColor: 'rgba(7, 12, 22, 0.86)',
        boxShadow: '0 10px 36px rgba(0, 0, 0, 0.35)',
      }}
    >
      {children}
    </div>
  );
};

const FlagPill = ({label, value}: {label: string; value: boolean | undefined}) => {
  return (
    <div
      style={{
        display: 'flex',
        alignItems: 'center',
        gap: 8,
        padding: '7px 10px',
        borderRadius: 999,
        border: `1px solid ${statusColor(value)}`,
        backgroundColor: `${statusColor(value)}20`,
        fontSize: 14,
      }}
    >
      <span style={{fontWeight: 700}}>{label}</span>
      <span>{chipText(value)}</span>
    </div>
  );
};

const PredictionBlock = ({prediction}: {prediction?: Record<string, unknown>}) => {
  if (!prediction || Object.keys(prediction).length === 0) {
    return (
      <div style={{fontFamily: 'Menlo, Consolas, monospace', fontSize: 14, color: '#ffcc88'}}>
        No prediction JSON
      </div>
    );
  }

  const entries = Object.entries(prediction).slice(0, 10);

  return (
    <div
      style={{
        fontFamily: 'Menlo, Consolas, monospace',
        fontSize: 14,
        lineHeight: 1.5,
        color: '#e8f2ff',
        whiteSpace: 'pre-wrap',
      }}
    >
      {'{'}
      {entries.map(([key, value], index) => {
        const valueText =
          typeof value === 'string' ? `"${value}"` : typeof value === 'object' ? '[object]' : String(value);

        return (
          <div key={`${key}-${index}`} style={{paddingLeft: 14}}>
            <span style={{color: '#8cd7ff'}}>{`"${key}"`}</span>
            <span style={{color: '#dce9ff'}}>{': '}</span>
            <span style={{color: '#ffd79c'}}>{valueText}</span>
            {index < entries.length - 1 ? ',' : ''}
          </div>
        );
      })}
      <div>{'}'}</div>
    </div>
  );
};

const IntroScene = ({
  frame,
  title,
  subtitle,
  episode,
}: {
  frame: number;
  title: string;
  subtitle?: string;
  episode: EpisodeResult | null;
}) => {
  const opacity = interpolate(frame, [0, 16], [0, 1], clampConfig);
  const y = interpolate(frame, [0, 22], [28, 0], clampConfig);

  return (
    <AbsoluteFill style={{justifyContent: 'center', alignItems: 'center', opacity}}>
      <Panel>
        <div style={{padding: '30px 44px', width: 1220, transform: `translateY(${y}px)`}}>
          <div style={{fontSize: 58, fontFamily: 'Impact, Haettenschweiler, sans-serif', color: '#f8f3de'}}>
            {title}
          </div>
          <div style={{fontSize: 24, marginTop: 8, color: '#cfe4ff'}}>{subtitle ?? 'Rollout comparison'}</div>
          <div
            style={{
              marginTop: 24,
              display: 'grid',
              gridTemplateColumns: '1fr 1fr',
              gap: 14,
              fontSize: 22,
            }}
          >
            <div>
              <div>Episode: {episode?.episodeId ?? 'N/A'}</div>
              <div>Track: {episode?.track ?? 'N/A'}</div>
            </div>
            <div>
              <div>Difficulty: {episode?.difficulty ?? '-'}</div>
              <div>Attack Family: {episode?.attackFamily ?? 'N/A'}</div>
            </div>
          </div>
          <div style={{marginTop: 20, fontSize: 20, color: '#f7d7aa'}}>
            Attacker goal: {episode?.attackerGoal ?? 'No attacker goal available for selected episode.'}
          </div>
        </div>
      </Panel>
    </AbsoluteFill>
  );
};

const LanesScene = ({
  frame,
  episode,
}: {
  frame: number;
  episode: EpisodeResult | null;
}) => {
  const laneReveal = (delay: number) => interpolate(frame, [delay, delay + 22], [0, 1], clampConfig);

  return (
    <AbsoluteFill style={{padding: 26, gap: 14}}>
      <div
        style={{
          fontSize: 34,
          fontFamily: 'Impact, Haettenschweiler, sans-serif',
          color: '#f6f0d3',
          letterSpacing: 1,
        }}
      >
        Episode Narrative Lanes
      </div>
      <div style={{display: 'grid', gridTemplateRows: '1fr 1fr 1fr', gap: 12, flex: 1}}>
        <Panel>
          <div style={{padding: 18, opacity: laneReveal(0)}}>
            <div style={{fontSize: 26, color: '#ff9ca8', fontWeight: 700}}>Attacker Lane</div>
            <div style={{fontSize: 20, marginTop: 10, color: '#ffdfdf'}}>
              Attack family: {episode?.attackFamily ?? 'N/A'}
            </div>
            <div style={{fontSize: 18, marginTop: 8, color: '#f4ccd2'}}>
              {episode?.attackerGoal ?? 'No attacker goal text available.'}
            </div>
          </div>
        </Panel>

        <Panel>
          <div style={{padding: 18, opacity: laneReveal(16)}}>
            <div style={{fontSize: 26, color: '#8ac8ff', fontWeight: 700}}>Defender Lane</div>
            <div style={{fontSize: 18, marginTop: 10, color: '#d9ecff'}}>
              Intended target: {episode?.oversightTarget ?? 'Not provided'}
            </div>
            <div style={{fontSize: 18, marginTop: 8, color: '#d8ebff'}}>
              Actual behavior: {episode?.defenderBehaviorSummary ?? 'Not provided'}
            </div>
          </div>
        </Panel>

        <Panel>
          <div style={{padding: 18, opacity: laneReveal(32)}}>
            <div style={{fontSize: 26, color: '#ffd37d', fontWeight: 700}}>Overseer Lane</div>
            <div style={{display: 'flex', gap: 8, flexWrap: 'wrap', marginTop: 10}}>
              <FlagPill label="Attack" value={episode?.flags.attackCorrect} />
              <FlagPill label="Failure" value={episode?.flags.failureCorrect} />
              <FlagPill label="Culprit" value={episode?.flags.culpritExact} />
              <FlagPill label="Schema" value={episode?.flags.schemaValid} />
            </div>
            <div style={{marginTop: 12}}>
              <PredictionBlock prediction={episode?.prediction} />
            </div>
          </div>
        </Panel>
      </div>
    </AbsoluteFill>
  );
};

const ScoreboardScene = ({
  frame,
  episode,
}: {
  frame: number;
  episode: EpisodeResult | null;
}) => {
  const components = (episode?.rewardComponents ? Object.entries(episode.rewardComponents) : [])
    .sort((a, b) => metricOrder.indexOf(a[0]) - metricOrder.indexOf(b[0]))
    .filter(([key]) => metricOrder.includes(key));

  const maxAbs = Math.max(1, ...components.map(([, value]) => Math.abs(value)));

  return (
    <AbsoluteFill style={{padding: 28}}>
      <Panel>
        <div style={{padding: 24, height: '100%', display: 'flex', flexDirection: 'column'}}>
          <div
            style={{
              fontSize: 36,
              fontFamily: 'Impact, Haettenschweiler, sans-serif',
              color: '#f7efcf',
            }}
          >
            Reward Component Scoreboard
          </div>
          {components.length === 0 ? (
            <div style={{marginTop: 20, fontSize: 18, color: '#ffd6a5'}}>
              No reward component breakdown available for this episode.
            </div>
          ) : (
            <div
              style={{
                marginTop: 14,
                display: 'grid',
                gridTemplateRows: `repeat(${components.length}, 1fr)`,
                gap: 8,
                flex: 1,
              }}
            >
              {components.map(([key, value], index) => {
              const reveal = interpolate(frame, [index * 8, index * 8 + 20], [0, 1], clampConfig);
              const width = Math.max(0, (Math.abs(value) / maxAbs) * 320 * reveal);
              const positive = value >= 0;

              return (
                <div
                  key={key}
                  style={{
                    display: 'grid',
                    gridTemplateColumns: '260px 1fr 100px',
                    alignItems: 'center',
                    gap: 14,
                  }}
                >
                  <div style={{fontSize: 17, color: '#d8e8ff'}}>{formatLabel(key)}</div>
                  <div style={{position: 'relative', height: 18, borderRadius: 999, backgroundColor: 'rgba(22, 33, 53, 0.85)'}}>
                    <div
                      style={{
                        position: 'absolute',
                        left: '50%',
                        top: 0,
                        width: 2,
                        height: '100%',
                        backgroundColor: 'rgba(220, 230, 255, 0.8)',
                      }}
                    />
                    <div
                      style={{
                        position: 'absolute',
                        top: 2,
                        height: 14,
                        borderRadius: 999,
                        left: positive ? '50%' : `calc(50% - ${width}px)`,
                        width,
                        backgroundColor: positive ? '#7fffbc' : '#ff8c8c',
                      }}
                    />
                  </div>
                  <div style={{textAlign: 'right', fontFamily: 'Menlo, Consolas, monospace', color: '#f3f8ff'}}>
                    {value.toFixed(2)}
                  </div>
                </div>
              );
              })}
            </div>
          )}
          <div style={{marginTop: 18, fontSize: 26, color: '#ffe8b0'}}>
            Total Reward: {episode ? episode.reward.toFixed(2) : '-'}
          </div>
        </div>
      </Panel>
    </AbsoluteFill>
  );
};

const RunCard = ({run, episode}: {run: RunSummary; episode: EpisodeResult | null}) => {
  return (
    <Panel>
      <div style={{padding: 18, display: 'flex', flexDirection: 'column', gap: 8}}>
        <div style={{fontSize: 28, color: '#f7f2d4', fontFamily: 'Impact, Haettenschweiler, sans-serif'}}>
          {run.name}
        </div>
        <div style={{fontSize: 17, color: '#d8e8ff'}}>Mean Reward: {formatScore(run.metrics.meanReward)}</div>
        <div style={{fontSize: 17, color: '#d8e8ff'}}>Culprit Exact Rate: {formatPercent(run.metrics.culpritExactRate)}</div>
        <div style={{fontSize: 17, color: '#d8e8ff'}}>Invalid Action Rate: {formatPercent(run.metrics.invalidActionRate)}</div>
        <div style={{fontSize: 17, color: '#d8e8ff'}}>JSON Only Rate: {formatPercent(run.metrics.jsonOnlyRate)}</div>
        <div style={{fontSize: 17, color: '#d8e8ff'}}>Schema Valid Rate: {formatPercent(run.metrics.schemaValidRate)}</div>
        <div style={{marginTop: 6, fontSize: 15, color: '#f3e0b4'}}>
          Episode Reward: {episode ? episode.reward.toFixed(2) : '-'}
        </div>
      </div>
    </Panel>
  );
};

const CompareScene = ({
  runs,
  selectedEpisodeId,
}: {
  runs: RunSummary[];
  selectedEpisodeId: string | null;
}) => {
  return (
    <AbsoluteFill style={{padding: 28, gap: 14}}>
      <div style={{fontSize: 36, color: '#f7efcf', fontFamily: 'Impact, Haettenschweiler, sans-serif'}}>
        Model Battle Compare
      </div>
      <div style={{display: 'grid', gridTemplateColumns: `repeat(${runs.length}, 1fr)`, gap: 14, flex: 1}}>
        {runs.map((run) => (
          <RunCard key={run.name} run={run} episode={getEpisodeForRun(run, selectedEpisodeId)} />
        ))}
      </div>
    </AbsoluteFill>
  );
};

const OutroScene = ({runs}: {runs: RunSummary[]}) => {
  const bestRun = [...runs]
    .sort((a, b) => (b.metrics.meanReward ?? -Infinity) - (a.metrics.meanReward ?? -Infinity))[0]
    ?.name;

  return (
    <AbsoluteFill style={{justifyContent: 'center', alignItems: 'center'}}>
      <Panel>
        <div style={{padding: '28px 40px', textAlign: 'center'}}>
          <div style={{fontSize: 52, color: '#f6f1d7', fontFamily: 'Impact, Haettenschweiler, sans-serif'}}>
            Final Score Panel
          </div>
          <div style={{marginTop: 12, fontSize: 26, color: '#d6e9ff'}}>
            Best mean reward run: {bestRun ?? 'N/A'}
          </div>
          <div style={{marginTop: 8, fontSize: 20, color: '#f2d8a8'}}>
            Ready for screenshot export.
          </div>
        </div>
      </Panel>
    </AbsoluteFill>
  );
};

export const OverseerRolloutVizComposition = (props: OverseerVizInput) => {
  const frame = useCurrentFrame();

  const resolved = resolveOverseerInput(props);
  const ranges = getOverseerSceneRanges();

  return (
    <AbsoluteFill
      style={{
        background:
          'radial-gradient(circle at 20% 10%, rgba(43, 69, 105, 0.55), rgba(7, 10, 19, 0.95) 55%), linear-gradient(145deg, #070b14, #0b1221)',
        color: '#eaf4ff',
        fontFamily: 'Trebuchet MS, Tahoma, Verdana, sans-serif',
      }}
    >
      <Audio src={staticFile('assets/audio/music/8BitBattleLoop.ogg')} loop volume={0.1} />

      <Sequence from={ranges.introStart} durationInFrames={sceneFrames.intro}>
        <IntroScene
          frame={frame - ranges.introStart}
          title={props.title}
          subtitle={props.subtitle}
          episode={resolved.primaryEpisode}
        />
      </Sequence>

      <Sequence from={ranges.lanesStart} durationInFrames={sceneFrames.lanes}>
        <LanesScene frame={frame - ranges.lanesStart} episode={resolved.primaryEpisode} />
      </Sequence>

      <Sequence from={ranges.scoreStart} durationInFrames={sceneFrames.scoreboard}>
        <ScoreboardScene frame={frame - ranges.scoreStart} episode={resolved.primaryEpisode} />
      </Sequence>

      <Sequence from={ranges.compareStart} durationInFrames={sceneFrames.compare}>
        <CompareScene runs={resolved.runs} selectedEpisodeId={resolved.selectedEpisodeId} />
      </Sequence>

      <Sequence from={ranges.outroStart} durationInFrames={sceneFrames.outro}>
        <OutroScene runs={resolved.runs} />
      </Sequence>
    </AbsoluteFill>
  );
};
