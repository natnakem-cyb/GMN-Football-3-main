/**
 * GMN-Football-3 — Scenario Completion Logic Regression Test
 *
 * Verifies the competitive match objective fixes for 5v5 / 11v11:
 *   - win_match requires a real scoreboard LEAD at match end (a draw, or
 *     conceding, is NOT a win), not merely "left team scored once".
 *   - control_possession (5v5) is resolved from final possession > 50%.
 *   - clean_sheet (11v11) is resolved from conceding 0 goals.
 *   - scenario_complete / scenario_failed are emitted exactly once per episode.
 *   - Drill objectives (score_goal, within_time) remain live/unchanged.
 */
import { GameEngine } from '../src/engine/GameEngine';
import { ACADEMY_SCENARIOS } from '../src/scenarios/ScenarioRegistry';

let failures = 0;
function check(cond: boolean, label: string) {
  if (cond) console.log(`   ✓ ${label}`);
  else {
    failures++;
    console.error(`   ✗ ${label}`);
  }
}

function runMatch(scenarioId: string, cfg: { left: number; right: number; possessionLeft: number }): GameEngine {
  const sc = ACADEMY_SCENARIOS.find((s) => s.id === scenarioId)!;
  const e = new GameEngine();
  e.loadScenario(sc, 7);
  e.score = { left: cfg.left, right: cfg.right };
  e.stats.possession.left = cfg.possessionLeft;
  e.stats.possession.right = 100 - cfg.possessionLeft;
  e.matchTimeSeconds = sc.timeLimitSeconds; // next step crosses the time limit
  e.step(new Map(), 1 / 60);
  return e;
}

function objective(e: GameEngine, id: string) {
  return e.activeScenario!.objectives.find((o) => o.id === id)!;
}

function resolutionEvents(e: GameEngine): { complete: number; failed: number } {
  return {
    complete: e.events.filter((ev) => ev.type === 'scenario_complete').length,
    failed: e.events.filter((ev) => ev.type === 'scenario_failed').length,
  };
}

function test5v5(): void {
  console.log('\n[5v5] win_match / control_possession');
  // Win: leads 1-0 with majority possession.
  const win = runMatch('5_vs_5', { left: 1, right: 0, possessionLeft: 60 });
  check(objective(win, 'win_match').isCompleted, '1-0 lead + NO opponent goals -> win_match COMPLETED');
  check(objective(win, 'win_match').isFailed === false, 'win_match not marked failed on a win');
  check(objective(win, 'control_possession').isCompleted, 'possession 60% -> control_possession COMPLETED');
  let r = resolutionEvents(win);
  check(r.complete === 1 && r.failed === 0, 'win emits exactly one scenario_complete');

  // Exactly once: stepping again must not re-emit.
  win.step(new Map(), 1 / 60);
  r = resolutionEvents(win);
  check(r.complete === 1, `scenario_complete emitted only once (got ${r.complete})`);

  // Loss: concedes 2, low possession.
  const loss = runMatch('5_vs_5', { left: 1, right: 2, possessionLeft: 30 });
  check(objective(loss, 'win_match').isFailed, '1-2 (losing) -> win_match FAILED (must LEAD, not merely score)');
  check(objective(loss, 'control_possession').isFailed, 'possession 30% -> control_possession FAILED');
  r = resolutionEvents(loss);
  check(r.failed === 1 && r.complete === 0, 'loss emits exactly one scenario_failed');

  // Draw: leading is required, a draw is not a win.
  const draw = runMatch('5_vs_5', { left: 0, right: 0, possessionLeft: 55 });
  check(objective(draw, 'win_match').isFailed, '0-0 draw -> win_match FAILED (draw is not a win)');
}

function test11v11(): void {
  console.log('\n[11v11] win_match / clean_sheet');
  const win = runMatch('11_vs_11', { left: 1, right: 0, possessionLeft: 55 });
  check(objective(win, 'win_match').isCompleted, '1-0 -> win_match COMPLETED');
  check(objective(win, 'clean_sheet').isCompleted, 'conceded 0 -> clean_sheet COMPLETED');
  let r = resolutionEvents(win);
  check(r.complete === 1 && r.failed === 0, 'clean-sheet win emits exactly one scenario_complete');

  const draw = runMatch('11_vs_11', { left: 0, right: 0, possessionLeft: 55 });
  check(objective(draw, 'win_match').isFailed, '0-0 draw -> win_match FAILED (draw is not a win)');
  check(objective(draw, 'clean_sheet').isCompleted, '0-0 draw still keeps the clean sheet COMPLETED');
  r = resolutionEvents(draw);
  check(r.failed === 1, 'draw emits exactly one scenario_failed (win_match not met)');

  const loss = runMatch('11_vs_11', { left: 0, right: 1, possessionLeft: 55 });
  check(objective(loss, 'win_match').isFailed, '0-1 -> win_match FAILED');
  check(objective(loss, 'clean_sheet').isFailed, 'conceded 1 -> clean_sheet FAILED');
}

function testDrillObjectivesUnchanged(): void {
  console.log('\n[drill] score_goal / within_time unchanged');
  const sc = ACADEMY_SCENARIOS.find((s) => s.id === 'academy_empty_goal')!;
  const e = new GameEngine();
  e.loadScenario(sc, 7);
  e.score = { left: 1, right: 0 };
  e.matchTimeSeconds = 5; // well before the 15s limit -> within_time completes
  e.step(new Map(), 1 / 60);
  check(objective(e, 'score_goal').isCompleted, 'score_goal completes live when left scores');
  check(objective(e, 'within_time').isCompleted, 'within_time completes when goal scored before limit');
}

function main() {
  console.log('====================================================');
  console.log('GMN-FOOTBALL-3 — SCENARIO COMPLETION LOGIC TEST');
  console.log('====================================================');
  test5v5();
  test11v11();
  testDrillObjectivesUnchanged();
  console.log('\n====================================================');
  if (failures === 0) console.log('✓ ALL SCENARIO COMPLETION CHECKS PASSED');
  else {
    console.error(`✗ ${failures} SCENARIO COMPLETION CHECK(S) FAILED`);
    process.exit(1);
  }
  console.log('====================================================');
}

main();