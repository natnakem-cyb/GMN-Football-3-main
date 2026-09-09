/**
 * Debug script to trace event transmission in GameEngine.step()
 */
import { GameEngine } from '../src/engine/GameEngine';
import { ActionType } from '../src/types/football';

const scenario = {
  id: 'academy_3_vs_1_with_keeper',
  name: 'Academy: 3 vs 1 with Keeper',
  codeName: 'academy_3_vs_1_with_keeper',
  description: 'Test',
  instructions: 'Test',
  difficulty: 'Intermediate' as const,
  stage: 4,
  teamLeftPlayers: 3,
  teamRightPlayers: 1,
  hasGoalkeeperLeft: false,
  hasGoalkeeperRight: true,
  timeLimitSeconds: 30,
  setup: {
    ball: { x: 0.25, y: 0, z: 0 } as any,
    leftPlayers: [
      { role: 'ST' as any, pos: { x: 0.2, y: -0.3 } },
      { role: 'CM' as any, pos: { x: 0.15, y: 0.2 } },
      { role: 'LW' as any, pos: { x: 0.1, y: -0.1 } },
    ],
    rightPlayers: [
      { role: 'CB' as any, pos: { x: 0.6, y: 0 } },
      { role: 'GK' as any, pos: { x: 0.95, y: 0 } },
    ],
    positionJitter: 0,
  },
  objectives: [],
  terminateOnOpponentPossession: false,
  rewards: { scoring: 1.0, completion: 0 },
};

const engine = new GameEngine();
engine.loadScenario(scenario, 42);

console.log('Starting SHOT spam test...');
for (let step = 0; step < 20; step++) {
  const actionMap = new Map<string, any>();
  engine.players.forEach((player) => {
    if (player.team === 'left') {
      actionMap.set(player.id, { type: ActionType.SHOT });
    }
  });
  
  const result = engine.step(actionMap, 1 / 60);
  
  console.log(`Step ${step}: event=${result.info.event ?? 'undefined'}, reward=${result.reward.toFixed(4)}, score=${result.info.score.left}-${result.info.score.right}, ballX=${engine.ball.position.x.toFixed(4)}, owner=${engine.ball.ownerId ?? 'none'}`);
  
  if (result.terminated) {
    console.log(`Terminated at step ${step}`);
    break;
  }
}
