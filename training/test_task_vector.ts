import { GameEngine } from '../src/engine/GameEngine';
import { ACADEMY_SCENARIOS } from '../src/scenarios/ScenarioRegistry';
import { TaskEncoder, TASK_VECTOR_DIM } from '../src/engine/TaskEncoder';

console.log('====================================================');
console.log('GMN-FOOTBALL-3 — TASK VECTOR PHASE 2 VALIDATION');
console.log('====================================================');

let totalTests = 0;
let passedTests = 0;

function check(cond: boolean, label: string) {
  totalTests++;
  if (cond) {
    console.log(`  OK  ${label}`);
    passedTests++;
  } else {
    console.error(`  FAIL ${label}`);
  }
}

// --- Schema: fixed dimension ---
console.log('\n[Schema] Fixed vector dimension...');
try {
  const engine = new GameEngine();
  engine.loadScenario(ACADEMY_SCENARIOS[0], 12345);
  const state = engine.getScenarioDynamicState();
  const vec = TaskEncoder.encode(engine.activeScenario, state);
  check(vec.length === TASK_VECTOR_DIM, `TaskEncoder returns ${TASK_VECTOR_DIM}-dim vector`);
  check(Array.from(vec).every(v => Number.isFinite(v)), 'All entries are finite numbers');
} catch (err: any) {
  console.error('  FAIL schema:', err.message);
}

// --- Schema: all scenarios produce same-dimension vector ---
console.log('\n[Schema] All scenarios produce stable dimension...');
try {
  for (const scenario of ACADEMY_SCENARIOS) {
    const engine = new GameEngine();
    engine.loadScenario(scenario, 12345);
    const state = engine.getScenarioDynamicState();
    const vec = TaskEncoder.encode(engine.activeScenario, state);
    check(vec.length === TASK_VECTOR_DIM, `${scenario.id}: vector length is ${TASK_VECTOR_DIM}`);
  }
} catch (err: any) {
  console.error('  FAIL schema all-scenarios:', err.message);
}

// --- Determinism: same inputs -> same output ---
console.log('\n[Determinism] Same config + state -> identical vector...');
try {
  const scenario = ACADEMY_SCENARIOS.find(s => s.id === 'academy_3_vs_1_with_keeper')!;
  const engine1 = new GameEngine();
  engine1.loadScenario(scenario, 12345);
  const state1 = engine1.getScenarioDynamicState();
  const vec1 = TaskEncoder.encode(engine1.activeScenario, state1);

  const engine2 = new GameEngine();
  engine2.loadScenario(scenario, 12345);
  const state2 = engine2.getScenarioDynamicState();
  const vec2 = TaskEncoder.encode(engine2.activeScenario, state2);

  const match = vec1.every((v, i) => v === vec2[i]);
  check(match, 'Identical inputs produce identical vectors');
} catch (err: any) {
  console.error('  FAIL determinism:', err.message);
}

// --- Bounds: every feature in documented range ---
console.log('\n[Bounds] All features within documented range...');
try {
  const scenario = ACADEMY_SCENARIOS.find(s => s.id === 'academy_rondo_4v1')!;
  const engine = new GameEngine();
  engine.loadScenario(scenario, 12345);
  const emptyActions = new Map<string, any>();
  for (let i = 0; i < 10; i++) {
    engine.step(emptyActions, 1 / 60);
  }
  const state = engine.getScenarioDynamicState();
  const vec = TaskEncoder.encode(engine.activeScenario, state);

  check(vec[0] === 0.0 || vec[0] === 1.0, 'terminate_on_turnover in {0,1}');
  check(vec[1] === 0.0 || vec[1] === 1.0, 'spatial_bounds_active in {0,1}');
  check(vec[2] >= 0.0 && vec[2] <= 1.0, 'time_remaining_frac in [0,1]');
  check(vec[3] >= 0.0 && vec[3] <= 1.0, 'pass_progress in [0,1]');
  check(vec[4] >= 0.0 && vec[4] <= 1.0, 'goal_progress in [0,1]');
  check(vec[5] === 0.0 || vec[5] === 1.0, 'possession_left in {0,1}');
  check(vec[6] === 0.0, 'reserved_1 == 0.0');
  check(vec[7] === 0.0, 'reserved_2 == 0.0');
} catch (err: any) {
  console.error('  FAIL bounds:', err.message);
}

// --- Missing data: unavailable fields do not fabricate values ---
console.log('\n[Missing Data] Unavailable semantics produce documented missing values...');
try {
  const scenario = ACADEMY_SCENARIOS.find(s => s.id === 'academy_empty_goal')!;
  const engine = new GameEngine();
  engine.loadScenario(scenario, 12345);
  const state = engine.getScenarioDynamicState();
  const vec = TaskEncoder.encode(engine.activeScenario, state);

  check(vec[3] === 0.0, 'pass_progress == 0.0 when targetPassesCount is unavailable');
  check(vec[4] === 0.0, 'goal_progress == 0.0 when targetGoalsCount is unavailable');
} catch (err: any) {
  console.error('  FAIL missing data:', err.message);
}

// --- Pass semantics: completedPasses is episode-total, not sequence stage ---
console.log('\n[Pass Semantics] completedPasses reflects episode total, not sequence...');
try {
  const scenario = ACADEMY_SCENARIOS.find(s => s.id === 'academy_3_vs_1_with_keeper')!;
  const engine = new GameEngine();
  engine.loadScenario(scenario, 12345);
  const emptyActions = new Map<string, any>();
  for (let i = 0; i < 5; i++) {
    engine.step(emptyActions, 1 / 60);
  }
  const state = engine.getScenarioDynamicState();
  check(state.passesCompleted >= 0, `passesCompleted is non-negative episode-total (${state.passesCompleted})`);
  check(state.sequenceStage === undefined, 'sequenceStage is undefined (unavailable)');
} catch (err: any) {
  console.error('  FAIL pass semantics:', err.message);
}

// --- Touch semantics: no fabricated touch counts ---
console.log('\n[Touch Semantics] No fabricated touch counts in dynamic state...');
try {
  const scenario = ACADEMY_SCENARIOS.find(s => s.id === 'academy_rondo_4v1')!;
  const engine = new GameEngine();
  engine.loadScenario(scenario, 12345);
  const emptyActions = new Map<string, any>();
  for (let i = 0; i < 10; i++) {
    engine.step(emptyActions, 1 / 60);
  }
  const state = engine.getScenarioDynamicState();
  check(state.touchesInCurrentPossession === undefined, 'touchesInCurrentPossession is undefined (not fabricated)');
} catch (err: any) {
  console.error('  FAIL touch semantics:', err.message);
}

// --- Turnover semantics: task termination projects engine termination ---
console.log('\n[Turnover Semantics] terminate_on_turnover reflects existing engine rules...');
try {
  const academyScenario = ACADEMY_SCENARIOS.find(s => s.id === 'academy_3_vs_1_with_keeper')!;
  const rondoScenario = ACADEMY_SCENARIOS.find(s => s.id === 'academy_rondo_4v1')!;

  const academyEngine = new GameEngine();
  academyEngine.loadScenario(academyScenario, 12345);
  const academyState = academyEngine.getScenarioDynamicState();
  const academyVec = TaskEncoder.encode(academyEngine.activeScenario, academyState);

  const rondoEngine = new GameEngine();
  rondoEngine.loadScenario(rondoScenario, 12345);
  const rondoState = rondoEngine.getScenarioDynamicState();
  const rondoVec = TaskEncoder.encode(rondoEngine.activeScenario, rondoState);

  check(academyVec[0] === 1.0, 'academy_3_vs_1_with_keeper: terminate_on_turnover == 1.0');
  check(rondoVec[0] === 0.0, 'academy_rondo_4v1: terminate_on_turnover == 0.0');
} catch (err: any) {
  console.error('  FAIL turnover semantics:', err.message);
}

// --- Regression: rawVector remains 127-dim ---
console.log('\n[Regression] rawVector remains 127 floats...');
try {
  const scenario = ACADEMY_SCENARIOS.find(s => s.id === 'academy_3_vs_1_with_keeper')!;
  const engine = new GameEngine();
  engine.loadScenario(scenario, 12345);
  const obs = engine.getObservation();
  check(obs.rawVector.length === 127, `rawVector length is 127 (got ${obs.rawVector.length})`);
  check(obs.zScenario !== undefined, 'zScenario is present');
  check(obs.zScenario!.length === TASK_VECTOR_DIM, `zScenario length is ${TASK_VECTOR_DIM}`);
} catch (err: any) {
  console.error('  FAIL regression:', err.message);
}

// --- Regression: step() observation includes zScenario ---
console.log('\n[Regression] step() observation includes zScenario...');
try {
  const scenario = ACADEMY_SCENARIOS.find(s => s.id === 'academy_3_vs_1_with_keeper')!;
  const engine = new GameEngine();
  engine.loadScenario(scenario, 12345);
  const emptyActions = new Map<string, any>();
  const res = engine.step(emptyActions, 1 / 60);
  check(res.observation.rawVector.length === 127, `step() rawVector length is 127 (got ${res.observation.rawVector.length})`);
  check(res.observation.zScenario !== undefined, 'step() observation.zScenario is present');
  check(res.observation.zScenario!.length === TASK_VECTOR_DIM, `step() zScenario length is ${TASK_VECTOR_DIM}`);
} catch (err: any) {
  console.error('  FAIL step() regression:', err.message);
}

// --- Dynamic: zScenario changes across steps ---
console.log('\n[Dynamic] zScenario changes across steps...');
try {
  const scenario = ACADEMY_SCENARIOS.find(s => s.id === 'academy_3_vs_1_with_keeper')!;
  const engine = new GameEngine();
  engine.loadScenario(scenario, 12345);
  const emptyActions = new Map<string, any>();

  const res0 = engine.step(emptyActions, 1 / 60);
  const z0 = res0.observation.zScenario;

  for (let i = 0; i < 5; i++) {
    engine.step(emptyActions, 1 / 60);
  }
  const res5 = engine.step(emptyActions, 1 / 60);
  const z5 = res5.observation.zScenario;

  check(z0 !== undefined && z5 !== undefined, 'zScenario defined at tick 0 and tick 5+');
  if (z0 !== undefined && z5 !== undefined) {
    check(z0[2] > z5[2], 'time_remaining_frac decreases over time');
  }
} catch (err: any) {
  console.error('  FAIL dynamic:', err.message);
}

console.log('\n====================================================');
console.log(`Task Vector Validation Summary: ${passedTests}/${totalTests} passed`);
console.log('====================================================');

if (passedTests !== totalTests) {
  process.exit(1);
}
