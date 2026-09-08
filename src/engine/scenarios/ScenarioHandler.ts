import { GameEngine } from '../GameEngine';

/**
 * Hook interface for scenario-specific behavior inside the core GameEngine loop.
 *
 * The engine delegates these four extension points to the active handler
 * (if any).  Most scenarios have no handler and fall through to the base
 * logic exactly as before.
 */
export interface ScenarioHandler {
  /** Called once when a scenario is loaded / reset. */
  onReset(): void;

  /**
   * Per-tick scenario-specific state tracking.
   * Runs after the standard physics/possession updates.
   * prevBallX is the ball's x-position captured at the start of the tick,
   * before any physics or possession updates.
   */
  onStep(engine: GameEngine, dt: number, prevBallX: number): void;

  /** If true, the engine skips the standard goal-and-boundary check for this tick. */
  skipStandardGoalCheck(): boolean;

  /**
   * Replace the base reward with a scenario-specific one.
   * Returning the input value preserves default behavior.
   */
  computeReward(baseReward: number, engine: GameEngine): number;

  /** Extra termination conditions beyond the standard 'fulltime'/academy-goal checks. */
  checkExtraTermination(engine: GameEngine): boolean;
}
