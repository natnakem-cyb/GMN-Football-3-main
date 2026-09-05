/**
 * Test: verify TrainingTelemetryService.ingestSnapshot() does not fabricate
 * fallback values when no real data has been received.
 *
 * This is a permanent, committed artifact — not a throwaway script.
 */

import { TrainingTelemetryService } from '../src/engine/TrainingTelemetryService';

function test_no_fabricated_fallbacks() {
  const service = TrainingTelemetryService.getInstance();
  service.resetMetrics();

  // Ingest an empty snapshot with no real data
  service.ingestSnapshot({});

  const snapshots = service.snapshots;
  console.log('Snapshots after empty ingest:', JSON.stringify(snapshots, null, 2));

  const last = snapshots[snapshots.length - 1];
  if (!last) {
    console.error('FAIL: no snapshot was created');
    process.exit(1);
  }

  const fields: Array<{ key: keyof typeof last; name: string }> = [
    { key: 'policyLoss', name: 'policyLoss' },
    { key: 'valueLoss', name: 'valueLoss' },
    { key: 'entropy', name: 'entropy' },
    { key: 'approxKl', name: 'approxKl' },
    { key: 'clipFraction', name: 'clipFraction' },
    { key: 'learningRate', name: 'learningRate' },
    { key: 'gradNorm', name: 'gradNorm' },
    { key: 'rollingReward', name: 'rollingReward' },
    { key: 'goalRate', name: 'goalRate' },
  ];

  let failed = false;
  for (const f of fields) {
    const val = last[f.key];
    if (val !== null) {
      console.error(`FAIL: ${f.name} = ${val} (expected null, got fabricated fallback)`);
      failed = true;
    } else {
      console.log(`   [OK] ${f.name} = null (no fabricated fallback)`);
    }
  }

  if (failed) {
    console.error('\nFAIL: Some fields still have fabricated fallback values!');
    process.exit(1);
  }

  console.log('\nPASS: All metric fields are null when no real data is available.');
}

function test_prior_snapshot_preserved() {
  const service = TrainingTelemetryService.getInstance();
  service.resetMetrics();

  // Ingest a real snapshot first
  service.ingestSnapshot({
    step: 1000,
    policyLoss: 0.05,
    valueLoss: 0.02,
    entropy: 1.5,
    approxKl: 0.01,
    clipFraction: 0.1,
    learningRate: 0.0003,
    gradNorm: 0.5,
    rollingReward: 0.3,
    goalRate: 10.0,
  });

  // Now ingest a partial snapshot missing some fields
  service.ingestSnapshot({
    step: 1024,
    policyLoss: 0.06,
    // valueLoss missing — should preserve previous value
  });

  const snapshots = service.snapshots;
  const last = snapshots[snapshots.length - 1];

  console.log('\nAfter partial snapshot ingest:');
  console.log('  policyLoss:', last.policyLoss, '(expected 0.06 — new value)');
  console.log('  valueLoss:', last.valueLoss, '(expected 0.02 — preserved from prior)');
  console.log('  entropy:', last.entropy, '(expected 1.5 — preserved from prior)');

  if (last.policyLoss !== 0.06) {
    console.error('FAIL: policyLoss should be 0.06');
    process.exit(1);
  }
  if (last.valueLoss !== 0.02) {
    console.error('FAIL: valueLoss should preserve prior value 0.02');
    process.exit(1);
  }
  if (last.entropy !== 1.5) {
    console.error('FAIL: entropy should preserve prior value 1.5');
    process.exit(1);
  }

  console.log('PASS: Prior snapshot values are preserved when new snapshot is partial.');
}

function main() {
  console.log('=== TrainingTelemetryService No-Fabrication Test ===\n');
  test_no_fabricated_fallbacks();
  test_prior_snapshot_preserved();
  console.log('\nAll tests passed.');
}

main();
