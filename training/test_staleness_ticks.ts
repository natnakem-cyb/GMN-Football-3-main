/**
 * Staleness-tick regression check for TrainedPolicyAgent.
 *
 * Drives decide() through two full decisionInterval=3 cycles and asserts
 * the exact observed stalenessTicks sequence. A microtask drain after
 * every 3rd tick lets the async ONNX .then() callback reset the counter.
 *
 * Expected sequence: [0, 1, 2, 2, 0, 1, 2, 2, 0]
 *   index 0: initial state
 *   indices 1-2: cached ticks (incrementing)
 *   index 3: decision tick while inference in-flight
 *   index 4: after async .then() resets to 0
 */

import { TrainedPolicyAgent } from '../src/agents/TrainedPolicyAgent';
import { AgentDecisionContext } from '../src/agents/BaseAgent';

// Minimal mock context
const mockContext: AgentDecisionContext = {
  player: {
    id: 'left_1',
    team: 'left',
    position: { x: 0, y: 0 },
    velocity: { x: 0, y: 0 },
    role: 'ST',
  } as any,
  teammates: [],
  opponents: [],
  ball: { position: { x: 0, y: 0 }, velocity: { x: 0, y: 0 } } as any,
  allPlayers: [],
  teamSide: 'left',
  controlledPlayerId: 'left_1',
  matchTime: 0,
  gameMode: 0,
};

async function main() {
  const agent = new TrainedPolicyAgent('test');
  agent.decisionInterval = 3;

  // Mock session: actOnnx resolves immediately with a dummy action
  const mockSession: any = {
    inputNames: ['obs'],
    outputNames: ['action_logits'],
    run: async () => ({
      action_logits: { data: new Float32Array(19) },
    }),
  };
  agent.session = mockSession;
  agent.isOnnxSessionActive = true;

  const observed: number[] = [agent.stalenessTicks]; // initial state

  for (let tick = 1; tick <= 6; tick++) {
    agent.decide(mockContext);
    observed.push(agent.stalenessTicks);

    // After every 3rd tick, drain microtasks so the async .then() runs
    // and resets stalenessTicks back to 0.
    if (tick % 3 === 0) {
      await new Promise((resolve) => setTimeout(resolve, 0));
      observed.push(agent.stalenessTicks);
    }
  }

  const expected = [0, 1, 2, 2, 0, 1, 2, 2, 0];

  console.log('Observed stalenessTicks sequence (6 ticks, interval=3):');
  console.log(JSON.stringify(observed, null, 2));
  console.log(`Expected: ${JSON.stringify(expected)}`);

  if (JSON.stringify(observed) !== JSON.stringify(expected)) {
    console.error('FAIL: stalenessTicks sequence does not match expected pattern.');
    process.exit(1);
  }

  console.log('PASS: stalenessTicks correctly increments and resets across fixed-interval async inference.');
}

main().catch((err) => {
  console.error('Staleness check failed:', err);
  process.exit(1);
});
