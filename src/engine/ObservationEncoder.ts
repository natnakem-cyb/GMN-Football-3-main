import { Ball, GameMode, MatchScore, Player, RLObservation, TeamSide } from '../types/football';
import type { GameEngine } from './GameEngine';
import {
  OBSERVATION_DIM,
  OBSERVATION_SCHEMA_VERSION,
  BASE_OBSERVATION_DIM,
  ROLE_DIM,
  ROLE_VOCABULARY,
  CONTROLLED_TRAINING_TEAM,
  inferPlayerRole,
  validateObservationVector,
} from './Contract';

export {
  OBSERVATION_DIM,
  OBSERVATION_SCHEMA_VERSION,
  BASE_OBSERVATION_DIM,
  ROLE_DIM,
  inferPlayerRole,
  validateObservationVector,
};

export class ObservationEncoder {
  /**
   * Generates a Game Model Network(GMN) compatible SMM / Feature vector observation
   * with role differentiation (127 floats total):
   * - Offset 0 (len 22): Left team player (x, y) positions, 11 players
   * - Offset 22 (len 22): Left team player (x, y) movement direction
   * - Offset 44 (len 22): Right team player (x, y) positions
   * - Offset 66 (len 22): Right team player (x, y) movement direction
   * - Offset 88 (len 3): Ball (x, y, z) position
   * - Offset 91 (len 3): Ball (x, y, z) movement direction
   * - Offset 94 (len 3): Ball ownership, one-hot: [no-one, left, right]
   * - Offset 97 (len 11): Active player, one-hot over 11 players
   * - Offset 108 (len 7): game_mode, one-hot: [Normal, KickOff, GoalKick, FreeKick, Corner, ThrowIn, Penalty]
   * - Offset 115 (len 12): Agent's assigned role one-hot over ROLE_VOCABULARY:
   *   [GK, CB, LB, RB, CDM, CM, LM, RM, LW, RW, CAM, ST]
   * Total: 127 floats. Inactive player slots are set to -1.
   */
  static encode(
    players: Player[],
    ball: Ball,
    viewpointPlayerId: string | null,
    score: MatchScore,
    stepCount: number,
    maxSteps: number,
    gameMode: GameMode = GameMode.Normal
  ): RLObservation {
    const leftPlayers = players.filter((p) => p.team === 'left');
    const rightPlayers = players.filter((p) => p.team === 'right');

    const leftPositions = leftPlayers.map((p) => [p.position.x, p.position.y]);
    const leftVelocities = leftPlayers.map((p) => [p.velocity.x * 50, p.velocity.y * 50]);
    const rightPositions = rightPlayers.map((p) => [p.position.x, p.position.y]);
    const rightVelocities = rightPlayers.map((p) => [p.velocity.x * 50, p.velocity.y * 50]);

    let ballOwnedTeam: -1 | 0 | 1 = -1;
    let ballOwnedPlayer = -1;

    if (ball.ownerId) {
      const owner = players.find((p) => p.id === ball.ownerId);
      if (owner) {
        if (owner.team === 'left') {
          ballOwnedTeam = 0;
          ballOwnedPlayer = leftPlayers.findIndex((p) => p.id === owner.id);
        } else {
          ballOwnedTeam = 1;
          ballOwnedPlayer = rightPlayers.findIndex((p) => p.id === owner.id);
        }
      }
    }

    const activeIndex = viewpointPlayerId
      ? (() => {
          // BUG-6 fix: the previous code only searched leftPlayers, so any
          // right-team viewpoint (bots / ONNX snapshot opponents) produced an
          // all-zero active slice (violating the 11-one-hot invariant) while
          // the structured field silently reported 0 via Math.max(0, -1).
          // Search the viewpoint player's own team first, then fall back to
          // the other side so every valid id yields exactly one hot bit.
          const li = leftPlayers.findIndex((p) => p.id === viewpointPlayerId);
          if (li >= 0) return li;
          const ri = rightPlayers.findIndex((p) => p.id === viewpointPlayerId);
          if (ri >= 0) return ri;
          return -1;
        })()
      : leftPlayers.length > 0
        ? 0
        : -1;

    // Construct flat rawVector with exactly OBSERVATION_DIM (127) floats
    const rawVector: number[] = [];

    // 0..21 (Length 22): Left team player (x, y) positions, 11 players
    for (let i = 0; i < 11; i++) {
      if (i < leftPlayers.length) {
        rawVector.push(leftPlayers[i].position.x, leftPlayers[i].position.y);
      } else {
        rawVector.push(-1.0, -1.0);
      }
    }

    // 22..43 (Length 22): Left team player (x, y) movement direction, 11 players
    for (let i = 0; i < 11; i++) {
      if (i < leftPlayers.length) {
        rawVector.push(leftPlayers[i].velocity.x * 50, leftPlayers[i].velocity.y * 50);
      } else {
        rawVector.push(-1.0, -1.0);
      }
    }

    // 44..65 (Length 22): Right team player (x, y) positions, 11 players
    for (let i = 0; i < 11; i++) {
      if (i < rightPlayers.length) {
        rawVector.push(rightPlayers[i].position.x, rightPlayers[i].position.y);
      } else {
        rawVector.push(-1.0, -1.0);
      }
    }

    // 66..87 (Length 22): Right team player (x, y) movement direction, 11 players
    for (let i = 0; i < 11; i++) {
      if (i < rightPlayers.length) {
        rawVector.push(rightPlayers[i].velocity.x * 50, rightPlayers[i].velocity.y * 50);
      } else {
        rawVector.push(-1.0, -1.0);
      }
    }

    // 88..90 (Length 3): Ball (x, y, z) position
    rawVector.push(ball.position.x, ball.position.y, ball.position.z);

    // 91..93 (Length 3): Ball (x, y, z) movement direction
    rawVector.push(ball.velocity.x * 50, ball.velocity.y * 50, ball.velocity.z * 50);

    // 94..96 (Length 3): Ball ownership, one-hot: [no-one, left, right]
    rawVector.push(
      ballOwnedTeam === -1 ? 1.0 : 0.0,
      ballOwnedTeam === 0 ? 1.0 : 0.0,
      ballOwnedTeam === 1 ? 1.0 : 0.0
    );

    // 97..107 (Length 11): Viewpoint / Controlled player, one-hot over 11 players
    for (let i = 0; i < 11; i++) {
      rawVector.push(activeIndex === i ? 1.0 : 0.0);
    }

    // 108..114 (Length 7): game_mode, one-hot: [Normal, KickOff, GoalKick, FreeKick, Corner, ThrowIn, Penalty]
    const modeIndices: GameMode[] = [
      GameMode.Normal,
      GameMode.KickOff,
      GameMode.GoalKick,
      GameMode.FreeKick,
      GameMode.Corner,
      GameMode.ThrowIn,
      GameMode.Penalty,
    ];
    for (const mode of modeIndices) {
      rawVector.push(gameMode === mode ? 1.0 : 0.0);
    }

    // 115..126 (Length 12): Self Agent Role One-Hot over ROLE_VOCABULARY
    // Computed dynamically from match state so role features are never left zero-padded
    const viewpointPlayer = viewpointPlayerId ? players.find((p) => p.id === viewpointPlayerId) : (leftPlayers[0] || players[0] || null);
    const resolvedRole = inferPlayerRole(viewpointPlayer);
    for (let r = 0; r < ROLE_VOCABULARY.length; r++) {
      rawVector.push(resolvedRole === ROLE_VOCABULARY[r] ? 1.0 : 0.0);
    }

    const validation = validateObservationVector(rawVector);
    if (!validation.valid) {
      throw new Error(
        `[ObservationEncoder Contract Violation] ${validation.reason}`
      );
    }

    return {
      leftTeamPositions: leftPositions,
      leftTeamVelocities: leftVelocities,
      rightTeamPositions: rightPositions,
      rightTeamVelocities: rightVelocities,
      ballPosition: [ball.position.x, ball.position.y, ball.position.z],
      ballVelocity: [ball.velocity.x, ball.velocity.y, ball.velocity.z],
      ballOwnedTeam,
      ballOwnedPlayer,
      activePlayerIndex: Math.max(0, activeIndex),
      gameMode: gameMode as number,
      score: [score.left, score.right],
      stepsLeft: Math.max(0, maxSteps - stepCount),
      rawVector,
    };
  }

  /**
   * Reward shaping computation (Phase 11 — exploit-resistant contract):
   *
   * Goal scored: +2.0 (terminal)
   * Goal conceded: -1.0 (terminal)
   * Ball progress checkpoint: max(+0.02, deltaX * 0.2) per step, only on new high-water mark
   * Verified pass completion: +0.15 per successful pass
   *
   * NOTE: Shot-attempt bonuses were deliberately removed in Phase 11 to prevent
   * reward exploitation via action spam. The checkpoint reward is intentionally
   * unconditional on shot quality; if possession-gated checkpointing is required,
   * the caller must supply ball-owner context and the function signature must be
   * extended accordingly.
   */
  static computeReward(
    prevBallX: number,
    currBallX: number,
    goalScoredTeam: TeamSide | null,
    targetTeam: TeamSide = CONTROLLED_TRAINING_TEAM,
    maxBallProgressX?: number,
    passCompletedByTargetTeam = false,
  ): { reward: number; checkpoint: number; newMaxBallProgressX: number } {
    if (targetTeam !== CONTROLLED_TRAINING_TEAM) {
      throw new Error(
        `[GMN Reward Invariant Violation] computeReward targetTeam='${targetTeam}' but ` +
          `CONTROLLED_TRAINING_TEAM='${CONTROLLED_TRAINING_TEAM}'. The current reward ` +
          `shaping and bridge metrics are only valid for the left-controlled team. ` +
          `Right-team training requires an explicit reward-orientation redesign.`
      );
    }

    let reward = 0;
    let checkpoint = 0;
    let newMaxBallProgressX = maxBallProgressX !== undefined ? maxBallProgressX : prevBallX;

    // Terminal goal events dominate intermediate shaping.
    if (goalScoredTeam === CONTROLLED_TRAINING_TEAM) {
      reward += 2.0;
    } else if (goalScoredTeam) {
      reward -= 1.0;
    }

    // Reduced monotonic checkpoint reward: pays only on new high-water mark.
    const isLeftControlled = targetTeam === CONTROLLED_TRAINING_TEAM;
    if (isLeftControlled) {
      if (currBallX > newMaxBallProgressX) {
        const deltaX = currBallX - newMaxBallProgressX;
        if (deltaX > 0.005) {
          checkpoint = Math.min(0.02, deltaX * 0.2);
          reward += checkpoint;
        }
        newMaxBallProgressX = currBallX;
      }
    } else {
      if (currBallX < newMaxBallProgressX) {
        const deltaX = newMaxBallProgressX - currBallX;
        if (deltaX > 0.005) {
          checkpoint = Math.min(0.02, deltaX * 0.2);
          reward += checkpoint;
        }
        newMaxBallProgressX = currBallX;
      }
    }

    // Explicit pass-completion reward: encourages meaningful passing.
    if (passCompletedByTargetTeam) {
      reward += 0.15;
    }

    return { reward, checkpoint, newMaxBallProgressX };
  }

  /**
   * Isolated dense reward for the Rondo 4v1 keep-ball drill.
   * No goal-scoring logic, no terminal bonus — fully dense.
   *
   * Shared across all agents (4 attackers + 1 defender) so a single
   * MAPPO policy learns behaviors useful for both sides.
   *
   * Reward components:
   * - Attacker possession in drill area: +0.01 per tick
   * - Completed attacker-to-attacker pass: +0.1
   * - Defender interception/tackle: +0.2
   * - Consecutive possession retention bonuses at 5s/10s/15s/20s
   *
   * @param prevBallX previous ball x position (unused but kept for signature stability)
   * @param currBallX current ball x position
   * @param currBallY current ball y position
   * @param ballOwnerTeam team that currently possesses the ball, or null
   * @param lastPassTeam team that completed the most recent pass
   * @param lastPassCompleted whether a pass was completed this tick
   * @param defenderDistToBall current defender-to-ball distance
   * @param prevDefenderDistToBall previous defender-to-ball distance
   * @param drillRadius radius of the drill area around center pitch
   * @param consecutivePossessionTime seconds of continuous left-team possession
   */
  static computeRondoReward({
    prevBallX: _prevBallX,
    currBallX,
    currBallY,
    ballOwnerTeam,
    lastPassTeam,
    lastPassCompleted,
    defenderDistToBall,
    prevDefenderDistToBall,
    drillRadius = 0.35,
    consecutivePossessionTime = 0,
  }: {
    prevBallX: number;
    currBallX: number;
    currBallY: number;
    ballOwnerTeam: TeamSide | null;
    lastPassTeam: TeamSide | null;
    lastPassCompleted: boolean;
    defenderDistToBall: number;
    prevDefenderDistToBall: number;
    drillRadius?: number;
    consecutivePossessionTime?: number;
  }): { attackerReward: number; defenderReward: number } {
    let attackerReward = 0;
    let defenderReward = 0;

    // Attackers: small per-tick reward for possession inside the drill area
    if (ballOwnerTeam === 'left') {
      if (Math.abs(currBallX) <= drillRadius && Math.abs(currBallY) <= drillRadius) {
        attackerReward += 0.01;
      }
    }

    // Completed attacker pass bonus
    if (lastPassCompleted && lastPassTeam === 'left') {
      attackerReward += 0.1;
    }

    // Defender: reward for winning possession via interception/tackle
    if (ballOwnerTeam === 'right') {
      defenderReward += 0.2;

      // Shaping: reward defender for closing distance to ball
      const distDelta = prevDefenderDistToBall - defenderDistToBall;
      if (distDelta > 0) {
        defenderReward += Math.min(0.05, distDelta * 0.5);
      }
    }

    // Temporal possession-retention bonuses for the attacking side
    if (ballOwnerTeam === 'left' && consecutivePossessionTime > 0) {
      if (consecutivePossessionTime >= 20.0) {
        attackerReward += 1.0;
      } else if (consecutivePossessionTime >= 15.0) {
        attackerReward += 0.8;
      } else if (consecutivePossessionTime >= 10.0) {
        attackerReward += 0.5;
      } else if (consecutivePossessionTime >= 5.0) {
        attackerReward += 0.3;
      }
    }

    return { attackerReward, defenderReward };
  }

  /**
   * Build a 19-element action mask for the given player/engine state.
   * 1 = valid, 0 = invalid.
   *
   * Movement actions (indices 0-7) and IDLE (index 8) are always valid.
   * Ball-handling actions (LONG_PASS=9, HIGH_PASS=10, SHORT_PASS=11,
   * SHOT=12, DRIBBLE=17) are only valid when the player has possession.
   * TACKLE (16) is only valid when the player does NOT have possession
   * (the engine already enforces this, but the mask makes it explicit to
   * the agent so it does not waste an action slot).
   */
  static getActionMask(player: Player, engine: GameEngine): number[] {
    const hasPossession = player.hasBall || engine.ball.ownerId === player.id;
    const mask: number[] = new Array(19).fill(1);

    // Ball-handling actions require possession.
    const BALL_ACTIONS = new Set<number>([9, 10, 11, 12, 17]);
    // TACKLE requires NO possession.
    const TACKLE_INDEX = 16;

    for (let i = 0; i < mask.length; i++) {
      if (i >= 0 && i <= 8) {
        // Movement + IDLE always valid.
        continue;
      }
      if (i === TACKLE_INDEX) {
        mask[i] = hasPossession ? 0 : 1;
        continue;
      }
      if (BALL_ACTIONS.has(i)) {
        mask[i] = hasPossession ? 1 : 0;
        continue;
      }
      // Remaining actions (SPRINT=13, RELEASE_DIRECTION=14,
      // RELEASE_SPRINT=15, RELEASE_DRIBBLE=18) are always valid.
      mask[i] = 1;
    }

    return mask;
  }
}
