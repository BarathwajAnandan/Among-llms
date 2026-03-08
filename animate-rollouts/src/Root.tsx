import {Composition, type CalculateMetadataFunction} from 'remotion';
import {
  OverseerRolloutVizComposition,
  getOverseerDurationInFrames,
} from './compositions/OverseerRolloutViz';
import {RolloutWrestlingComposition} from './compositions/RolloutWrestling';
import {defaultMatchInput} from './data/default-match';
import {defaultOverseerVizInput} from './data/default-overseer-viz';
import {compileMatch} from './lib/timeline';
import type {OverseerVizInput} from './overseer/types';
import type {MatchInput} from './types';

const calculateMetadata: CalculateMetadataFunction<MatchInput> = ({props}) => {
  const compiled = compileMatch(props);

  return {
    durationInFrames: Math.max(compiled.totalFrames, 1),
  };
};

const calculateOverseerMetadata: CalculateMetadataFunction<OverseerVizInput> = () => {
  return {
    durationInFrames: getOverseerDurationInFrames(),
  };
};

export const RemotionRoot = () => {
  return (
    <>
      <Composition
        id="RolloutWrestling"
        component={RolloutWrestlingComposition}
        width={1920}
        height={1080}
        fps={30}
        durationInFrames={600}
        defaultProps={defaultMatchInput}
        calculateMetadata={calculateMetadata}
      />
      <Composition
        id="OverseerRolloutViz"
        component={OverseerRolloutVizComposition}
        width={1920}
        height={1080}
        fps={30}
        durationInFrames={890}
        defaultProps={defaultOverseerVizInput}
        calculateMetadata={calculateOverseerMetadata}
      />
    </>
  );
};
