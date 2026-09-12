import { GameEngine } from '../GameEngine';
import { ObservationEncoder } from '../ObservationEncoder';
import { ScenarioHandler } from './ScenarioHandler';
import { TeamSide } from '../../types/football';

/**
 * ScenarioHandler for academy_rondo_4v1 (4v1 keep-ball drill).
 *
 * Extracted from GameEngine.ts so the core engine carries zero rondo-specific
 * state or conditionals.
 */
export class RondoScenarioHandler implements ScenarioHandler {
  public static readonly SCENARIO_ID = 'academy_rondo_4v1';

  // --- rondo state previously on GameEngine ---------------------------------
  private defenderPossessionTime = 0;
  private lastPassCompleted = false;
  private lastPassTeam: TeamSide | null = null;
  private prevCompletedPassesLeft = 0;
  private prevDefenderDistToBall = 0;
  private currentDefenderDistToBall = 0;
  private consecutivePossessionTime = 0;
  private lastPossessionTeam: TeamSide | null = null;
  private ballOutOfAreaTime = 0;
  private prevBallX = 0;
  private lastDefenderReward = 0;
  private defenderJustWonPossessionThisTick = false;

  // --- ScenarioHandler -------------------------------------------------------
  onReset(): void {
    this.defenderPossessionTime = 0;
    this.lastPassCompleted = false;
    this.lastPassTeam = null;
    this.prevCompletedPassesLeft = 0;
    this.prevDefenderDistToBall = 0;
    this.consecutivePossessionTime = 0;
    this.lastPossessionTeam = null;
    this.ballOutOfAreaTime = 0;
    this.prevBallX = 0;
    this.lastDefenderReward = 0;
    this.defenderJustWonPossessionThisTick = false;
  }

  onStep(engine: GameEngine, dt: number, prevBallX: number): void {
    this.prevBallX = prevBallX;
    // Detect pass completion by left team this tick: compare against the
    // snapshot persisted from the PREVIOUS tick. A function-local snapshot of
    // the current counter can never be greater than itself, so it must live
    // as member state (BUG-1 fix: pass bonus was dead code).
    const completedPassesLeft = engine.stats.completedPasses.left;
    this.lastPassCompleted = completedPassesLeft > this.prevCompletedPassesLeft;
    this.lastPassTeam = this.lastPassCompleted ? 'left' : null;
    this.prevCompletedPassesLeft = completedPassesLeft;

    // Defender distance to ball
    const defender = engine.players.find((p) => p.team === 'right');
    let currentDefenderDistToBall = 0;
    if (defender) {
      const defenderPos = defender.position;
      const ballPos = engine.ball.position;
      currentDefenderDistToBall = Math.hypot(
        defenderPos.x - ballPos.x,
        defenderPos.y - ballPos.y
      );
      this.currentDefenderDistToBall = currentDefenderDistToBall;
    }

    // Defender possession timer
    if (engine.ball.ownerId) {
      const owner = engine.players.find((p) => p.id === engine.ball.ownerId);
      if (owner?.team === 'right') {
        this.defenderPossessionTime += dt;
      } else {
        this.defenderPossessionTime = 0;
      }
    } else {
      this.defenderPossessionTime = 0;
    }

    // Consecutive possession timer for retention bonus.
    // Capture the PRE-mutation value of lastPossessionTeam before onStep mutates
    // it for the current tick. BUG-2 fix: defender transition reward was dead
    // code because computeReward compared the freshly-mutated field against
    // itself; snapshot here so computeReward can read the genuine previous tick.
    const previousPossessionTeam = this.lastPossessionTeam;
    const currentPossessionTeam = engine.ball.ownerId
      ? engine.players.find((p) => p.id === engine.ball.ownerId)?.team ?? null
      : null;
    this.defenderJustWonPossessionThisTick =
      currentPossessionTeam === 'right' && previousPossessionTeam !== 'right';

    if (currentPossessionTeam === 'left') {
      if (this.lastPossessionTeam === 'left') {
        this.consecutivePossessionTime += dt;
      } else {
        this.consecutivePossessionTime = 0;
      }
      this.lastPossessionTeam = 'left';
      this.ballOutOfAreaTime = 0;
    } else {
      this.consecutivePossessionTime = 0;
      this.lastPossessionTeam = currentPossessionTeam;
      if (currentPossessionTeam === 'right') {
        this.ballOutOfAreaTime = 0;
      }
    }

    // Ball-out-of-drill-area timer for early termination
    const ballX = engine.ball.position.x;
    const ballY = engine.ball.position.y;
    const inDrillArea = Math.abs(ballX) <= 0.35 && Math.abs(ballY) <= 0.35;
    if (!inDrillArea) {
      this.ballOutOfAreaTime += dt;
    } else {
      this.ballOutOfAreaTime = 0;
    }
  }

  skipStandardGoalCheck(): boolean {
    return true; // rondo has no goal objective
  }

  computeReward(_baseReward: number, engine: GameEngine): number {
    // Suppress base academy goal-scoring shaping; use isolated dense reward.
    const ballOwnerTeam = engine.ball.ownerId
      ? engine.players.find((p) => p.id === engine.ball.ownerId)?.team ?? null
      : null;

    const { attackerReward, defenderReward } = ObservationEncoder.computeRondoReward({
      prevBallX: this.prevBallX,
      currBallX: engine.ball.position.x,
      currBallY: engine.ball.position.y,
      ballOwnerTeam,
      lastPassTeam: this.lastPassTeam,
      lastPassCompleted: this.lastPassCompleted,
      defenderDistToBall: this.currentDefenderDistToBall,
      prevDefenderDistToBall: this.prevDefenderDistToBall,
      drillRadius: 0.35,
      consecutivePossessionTime: this.consecutivePossessionTime,
      defenderJustWonPossession: this.defenderJustWonPossessionThisTick,
    });
    this.lastDefenderReward = defenderReward;
    this.prevDefenderDistToBall = this.currentDefenderDistToBall;
    return attackerReward;
  }

  checkExtraTermination(_engine: GameEngine): boolean {
    if (this.defenderPossessionTime >= 2.0) {
      return true;
    }
    if (this.ballOutOfAreaTime >= 1.5) {
      return true;
    }
    return false;
  }

  // --- helpers consumed by ObservationEncoder.computeRondoReward -------------
  getDefenderPossessionTime(): number {
    return this.defenderPossessionTime;
  }

  getLastPassCompleted(): boolean {
    return this.lastPassCompleted;
  }

  getLastPassTeam(): TeamSide | null {
    return this.lastPassTeam;
  }

  getPrevDefenderDistToBall(): number {
    return this.prevDefenderDistToBall;
  }

  getConsecutivePossessionTime(): number {
    return this.consecutivePossessionTime;
  }

  getLastDefenderReward(): number {
    return this.lastDefenderReward;
  }
}
