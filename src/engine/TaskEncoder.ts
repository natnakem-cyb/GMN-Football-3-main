import { ScenarioConfig, ScenarioDynamicState } from '../types/football';

export const TASK_VECTOR_DIM = 8;

export class TaskEncoder {
  public static encode(config: ScenarioConfig | null | undefined, dynamicState: ScenarioDynamicState): Float32Array {
    const vector = new Float32Array(TASK_VECTOR_DIM);

    const taskSpec = config?.taskSpec;
    const constraints = taskSpec;

    // Index 0: terminate_on_turnover
    const terminateOnTurnover = constraints?.terminateOnTurnover ?? config?.terminateOnOpponentPossession ?? false;
    vector[0] = terminateOnTurnover ? 1.0 : 0.0;

    // Index 1: spatial_bounds_active
    vector[1] = constraints?.spatialBounds ? 1.0 : 0.0;

    // Index 2: time_remaining_frac
    const totalTicks = dynamicState.totalTicks > 0 ? dynamicState.totalTicks : 1;
    vector[2] = Math.max(0.0, dynamicState.ticksRemaining / totalTicks);

    // Index 3: pass_progress
    const targetPasses = constraints?.targetPassesCount;
    if (targetPasses && targetPasses > 0) {
      vector[3] = Math.min(dynamicState.passesCompleted / targetPasses, 1.0);
    } else {
      vector[3] = 0.0;
    }

    // Index 4: goal_progress
    const targetGoals = constraints?.targetGoalsCount;
    if (targetGoals && targetGoals > 0) {
      vector[4] = Math.min(dynamicState.goalsCompleted / targetGoals, 1.0);
    } else {
      vector[4] = 0.0;
    }

    // Index 5: possession_left
    vector[5] = dynamicState.currentPossessionTeam === 'left' ? 1.0 : 0.0;

    // Index 6-7: reserved for future use
    vector[6] = 0.0;
    vector[7] = 0.0;

    return vector;
  }
}
