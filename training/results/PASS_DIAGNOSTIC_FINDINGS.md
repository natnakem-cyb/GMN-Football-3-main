# PASS Diagnostic Findings — F_act Extension

**Date:** 2026-09-18 07:25 UTC
**Run ID:** pass_diag_2026-09-18
**HEAD:** `b0b6a3835972`
**Checkpoint SHA256 (seed 42):** `ca8bef84972a4775...`
**Checkpoint timesteps:** 50176
**Scenario:** academy_3_vs_1_with_keeper_onball
**Seeds:** 123, 42, 7, 999
**Episodes per seed:** 20

## 1. Provenance

- **HEAD:** `b0b6a3835972`
- **Seed 123 checkpoint:** `training/models/mappo_academy_3_vs_1_with_keeper_seed123_50176.pt`
  - SHA256: `629a928805cec616...`
  - Timesteps: 50176
- **Seed 42 checkpoint:** `training/models/mappo_academy_3_vs_1_with_keeper_seed42_50176.pt`
  - SHA256: `ca8bef84972a4775...`
  - Timesteps: 50176
- **Seed 7 checkpoint:** `training/models/mappo_academy_3_vs_1_with_keeper_seed7_50176.pt`
  - SHA256: `38b32f4cbd902ba7...`
  - Timesteps: 50176
- **Seed 999 checkpoint:** `training/models/mappo_academy_3_vs_1_with_keeper_seed999_50176.pt`
  - SHA256: `a1c5eacf57438573...`
  - Timesteps: 50176
- **Scenario:** academy_3_vs_1_with_keeper_onball
- **Deterministic:** True

## 2. Protocol

- Native PASS only (SHORT_PASS, index 11)
- K = 1 (forced action applied for exactly one tick)
- Post-force window: 50 ticks
- No clampPassDirection
- No teammate scripting
- No reward/GAE/mask/network/spawn changes
- Production PASS semantics unchanged

## 3. Trigger Validity

- Valid forced PASS episodes: 80
- Invalid episodes (excluded): 0

## 4. PASS Initiation

- Forced PASS episodes: 80
- PASS initiation events: 80

## 5. PASS Trajectory

- Median angular error to nearest teammate: 0.706 rad (40.5°)
- Perpendicular distance to pass ray: median=0.2190, min=0.0110
- Min ball-to-teammate distance: median=0.1991, min=0.0668

## 6. Teammate Response

- Median teammate speed after force: 0.4884
- Fraction teammates with non-zero response: 97.59%

## 7. Completion / Event Evidence

- PASS_COMPLETED events seen: 2 / 80
- Ownership transitions at min-distance tick: 0

## 8. Rank 1 / Rank 2 / Rank 3 Assessment

- **Rank 1:** 1 episodes (1.2%)
- **Unclassified:** 79 episodes (98.8%)

## 9. Conclusion

PASS_COMPLETED events were observed in this diagnostic run.

This document diagnoses only. It does not propose or implement corrective overrides.

## 10. No Proposed Fix

This document is diagnostic-only. No corrective override is proposed or implemented.
