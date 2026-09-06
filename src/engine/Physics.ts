import { Ball, Player, TeamSide, Vector2D, Vector3D } from '../types/football';
import { PITCH } from './Rules';
import { Vec2, Vec3 } from './Vector';
import { SeededRNG } from './SeededRNG';

export class PhysicsEngine {
  /**
   * FIXED SIMULATION TIMESTEP (60 Hz).
   *
   * The physics integration below is a FIXED-TIMESTEP simulation: drag, friction
   * and velocity smoothing are applied as per-tick multiplicative factors, NOT as
   * continuous dt-scaled terms. The API accepts a `dt` parameter but only
   * dt === FIXED_DT (1/60) is supported and verified. The training bridge always
   * steps with dt = 1/60. Callers passing any other dt will get physically
   * incorrect (tick-rate-dependent) results — do not treat this API as
   * dt-independent. See `training/TELEMETRY_PROTOCOL.md` and section 16 of the
   * stabilization notes.
   */
  static readonly FIXED_DT = 1 / 60;
  static GRAVITY = 9.8;
  static AIR_DRAG = 0.988;
  static GROUND_FRICTION = 0.965;
  static BOUNCE_RESTITUTION = 0.65;
  static PLAYER_ACCELERATION = 0.21;
  static PLAYER_MAX_SPEED = 0.96;
  static SPRINT_MULTIPLIER = 1.35;
  static TACKLE_SPEED = 1.68;
  static BALL_CONTROL_DIST = 0.038;

  // Step ball simulation
  static updateBall(ball: Ball, players: Player[], dt: number): void {
    // Save previous position to trail
    if (ball.trail.length > 20) {
      ball.trail.shift();
    }
    ball.trail.push({ ...ball.position });

    // If attached to player with possession
    if (ball.ownerId) {
      const owner = players.find((p) => p.id === ball.ownerId);
      if (owner) {
        const offset = Vec2.fromAngle(owner.heading, 0.025);
        ball.position.x = owner.position.x + offset.x;
        ball.position.y = owner.position.y + offset.y;
        ball.position.z = 0;
        ball.velocity = {
          x: owner.velocity.x * PhysicsEngine.GROUND_FRICTION,
          y: owner.velocity.y * PhysicsEngine.GROUND_FRICTION,
          z: 0,
        };
        ball.isInAir = false;
        return;
      } else {
        ball.ownerId = null;
      }
    }

    // Free ball dynamics
    ball.position.x += ball.velocity.x * dt;
    ball.position.y += ball.velocity.y * dt;
    ball.position.z += ball.velocity.z * dt;

    // Apply gravity
    if (ball.position.z > 0 || ball.velocity.z !== 0) {
      ball.velocity.z -= PhysicsEngine.GRAVITY * dt;
      ball.velocity.x *= PhysicsEngine.AIR_DRAG;
      ball.velocity.y *= PhysicsEngine.AIR_DRAG;
      ball.isInAir = true;
    } else {
      ball.position.z = 0;
      ball.velocity.z = 0;
      ball.velocity.x *= PhysicsEngine.GROUND_FRICTION;
      ball.velocity.y *= PhysicsEngine.GROUND_FRICTION;
      ball.isInAir = false;
    }

    // Ground bounce
    if (ball.position.z <= 0) {
      ball.position.z = 0;
      if (Math.abs(ball.velocity.z) > 0.3) {
        ball.velocity.z = -ball.velocity.z * PhysicsEngine.BOUNCE_RESTITUTION;
        ball.velocity.x *= 0.92;
        ball.velocity.y *= 0.92;
      } else {
        ball.velocity.z = 0;
      }
    }

    // Stop negligible velocities
    if (Vec3.length(ball.velocity) < 0.012) {
      ball.velocity = { x: 0, y: 0, z: 0 };
    }
  }

  // Update single player state & physics
  static updatePlayer(player: Player, ball: Ball, dt: number): void {
    // Tackle state
    if (player.isTackling) {
      player.tackleCooldown -= 1;
      player.position.x += player.velocity.x * dt;
      player.position.y += player.velocity.y * dt;
      player.velocity.x *= 0.88;
      player.velocity.y *= 0.88;

      if (player.tackleCooldown <= 0) {
        player.isTackling = false;
      }
      return;
    }

    if (player.tackleCooldown > 0) {
      player.tackleCooldown -= 1;
    }

    // Target tracking / movement calculation
    const delta = Vec2.sub(player.targetPosition, player.position);
    const distToTarget = Vec2.length(delta);

    if (distToTarget > 0.005) {
      const dir = Vec2.normalize(delta);
      const targetHeading = Vec2.angle(dir);

      // Smooth heading interpolation
      player.heading = targetHeading;

      // Calculate desired speed
      let maxSpeed = (player.stats.speed / 100) * PhysicsEngine.PLAYER_MAX_SPEED;
      if (player.isSprinting && player.stamina > 10) {
        maxSpeed *= PhysicsEngine.SPRINT_MULTIPLIER;
        player.stamina = Math.max(0, player.stamina - 0.25);
      } else {
        player.stamina = Math.min(100, player.stamina + 0.1);
      }

      // Acceleration towards desired velocity
      const targetVel = Vec2.scale(dir, Math.min(maxSpeed, distToTarget * 12.0));
      player.velocity.x = player.velocity.x * 0.75 + targetVel.x * 0.25;
      player.velocity.y = player.velocity.y * 0.75 + targetVel.y * 0.25;
    } else {
      player.velocity.x *= 0.7;
      player.velocity.y *= 0.7;
      player.stamina = Math.min(100, player.stamina + 0.15);
    }

    // Apply movement
    player.position.x += player.velocity.x * dt;
    player.position.y += player.velocity.y * dt;

    // Pitch boundary clamping for players
    player.position.x = Math.max(PITCH.minX - 0.05, Math.min(PITCH.maxX + 0.05, player.position.x));
    player.position.y = Math.max(PITCH.minY - 0.05, Math.min(PITCH.maxY + 0.05, player.position.y));
  }

  /**
   * Computes the ball velocity a kick produces, given a player, direction, power
   * and loft. Single source of truth for kick speed semantics — used by
   * `kickBall` and by the SHOT on-target projection so that shot-quality checks
   * simulate exactly the velocity the ball will actually receive.
   */
  static computeKickVelocity(
    player: Player,
    direction: Vector2D,
    power: number,
    loft = 0
  ): Vector3D {
    const dir = Vec2.normalize(direction);
    const kickSpeed = 0.9 + (player.stats.kickPower / 100) * 2.7 * Math.min(1, Math.max(0.1, power));
    return {
      x: dir.x * kickSpeed,
      y: dir.y * kickSpeed,
      z: loft * 1.32 * power,
    };
  }

  /**
   * Authoritative ballistic projection of a ball trajectory onto a goal-line
   * plane x = goalX. Simulates the same discrete per-tick integration used by
   * `updateBall` (air drag when airborne, ground friction when grounded, gravity,
   * bounces) at the FIXED_DT timestep, returning the (y, z) point at which the
   * ball first reaches the goal-line plane — or null if it never reaches it
   * within `maxTicks` (i.e. the shot cannot reach the goal).
   *
   * Used by goal-boundary-aware shot-quality logic (see `isGoalMouthPoint`).
   */
  static projectShotAtGoalLine(
    position: { x: number; y: number; z: number },
    velocity: { x: number; y: number; z: number },
    goalX: number,
    maxTicks = 600
  ): { y: number; z: number } | null {
    const dt = PhysicsEngine.FIXED_DT;
    let { x, y, z } = position;
    let vx = velocity.x;
    let vy = velocity.y;
    let vz = velocity.z;
    const travelingTowardLine = (goalX - x) * vx > 0;
    if (!travelingTowardLine && Math.abs(vx) < 1e-9) return null;

    for (let tick = 0; tick < maxTicks; tick++) {
      const prevX = x;
      // Horizontal integration (matches updateBall: drag only while airborne,
      // ground friction while grounded).
      const airborne = z > 0 || vz !== 0;
      if (airborne) {
        vx *= PhysicsEngine.AIR_DRAG;
        vy *= PhysicsEngine.AIR_DRAG;
      } else {
        vx *= PhysicsEngine.GROUND_FRICTION;
        vy *= PhysicsEngine.GROUND_FRICTION;
      }
      x += vx * dt;
      y += vy * dt;

      // Vertical integration (matches updateBall gravity + bounce semantics).
      if (z > 0 || vz !== 0) {
        vz -= PhysicsEngine.GRAVITY * dt;
      }
      z += vz * dt;
      if (z <= 0) {
        z = 0;
        if (Math.abs(vz) > 0.3) {
          vz = -vz * PhysicsEngine.BOUNCE_RESTITUTION;
        } else {
          vz = 0;
        }
      }

      // Goal-line plane crossing this tick?
      const crossed =
        (prevX < goalX && x >= goalX) || (prevX > goalX && x <= goalX) || x === goalX;
      if (crossed) {
        // Linear interpolation of y at the exact crossing point within the tick.
        const span = x - prevX;
        const frac = span !== 0 ? (goalX - prevX) / span : 1;
        const yAtLine = y - vy * dt * (1 - frac);
        return { y: yAtLine, z };
      }
    }
    return null;
  }

  // Kick ball in direction with power and loft
  static kickBall(
    ball: Ball,
    player: Player,
    direction: Vector2D,
    power: number,
    loft = 0
  ): void {
    ball.ownerId = null;
    player.hasBall = false;
    ball.lastOwnerId = player.id;
    ball.lastOwnerTeam = player.team;

    ball.velocity = PhysicsEngine.computeKickVelocity(player, direction, power, loft);
  }

  // Slide tackle
  static executeTackle(
    player: Player,
    ball: Ball,
    otherPlayers: Player[],
    rng?: SeededRNG
  ): 'success' | 'miss' | 'foul' {
    if (player.tackleCooldown > 0) return 'miss';

    player.isTackling = true;
    player.tackleCooldown = 25; // ticks

    const tackleDir = Vec2.fromAngle(player.heading, PhysicsEngine.TACKLE_SPEED);
    player.velocity = { x: tackleDir.x, y: tackleDir.y };

    // Check if close to ball owner or ball
    const ballPos2D = { x: ball.position.x, y: ball.position.y };
    const distToBall = Vec2.distance(player.position, ballPos2D);

    if (distToBall < 0.065) {
      if (ball.ownerId) {
        const owner = otherPlayers.find((p) => p.id === ball.ownerId);
        if (owner && owner.team !== player.team) {
          // Deterministic roll for foul chance (15% mistimed-tackle chance)
          // Uses seeded RNG if available, or spatial pseudo-hash
          const roll = rng
            ? rng.next()
            : (Math.abs(
                Math.sin(player.position.x * 12.9898 + player.position.y * 78.233 + distToBall * 43758.5453) *
                  43758.5453
              ) % 1);

          if (roll < 0.15) {
            // Foul committed: ball stays with original owner (no dispossession)
            return 'foul';
          }

          // Clean Dispossession
          owner.hasBall = false;
          ball.ownerId = null;
          ball.lastOwnerId = player.id;
          ball.lastOwnerTeam = player.team;

          // Push ball away or claim it (using deterministic deflection angle)
          const angleOffset = rng
            ? rng.nextRange(-0.2, 0.2)
            : (((roll * 10) % 1) - 0.5) * 0.4;
          const kickDir = Vec2.fromAngle(player.heading + angleOffset);
          ball.velocity = {
            x: kickDir.x * 1.2,
            y: kickDir.y * 1.2,
            z: 0.3,
          };
          return 'success';
        }
      }
    }
    return 'miss';
  }
}
