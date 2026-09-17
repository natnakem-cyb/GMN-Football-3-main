# Experimental Hierarchy

## Locked sequence

| Label | Name | Freeze / settings | Scope | Status |
|-------|------|-------------------|-------|--------|
| **A** | Frozen post-strip reference | Post-strip reward implementation, `E=1/β=0.03`, 100k timesteps | Reference baseline | Done |
| **B** | Horizon extension | Same freeze as A | 200k–500k timesteps | Blocked until D resolves structural issues |
| **C** | Controlled exploration ablation | Fixed horizon (100k), isolate exploration bonus `E` | Compare `(E=0, β=0)` vs `(E=1, β=0.03)` | Done |
| **D** | Targeted investigation | No training; probe-only repairs on frozen engine | Diagnose and fix action masking / execution-path issues before any further experiment | In progress |

## Dependencies and gating

- **A** is the locked reference. Do not retrain A.
- **B** must not start until the primary case is qualified and the structural issues identified in C/D are resolved.
- **C** was the characterization step for the exploration axis. Its finding was that exploration at `(E=1, β=0.03)` does not improve productive football behavior relative to `(E=0, β=0)` at 100k; the dominant pathologies are structural (masking, possession collapse, reward attribution), not horizon-related.
- **D** is the targeted repair track. Its exit criterion is primary-case scenario qualification (`F ∧ L ∧ C ∧ I ∧ E` all pass). Only after D exits may B be considered.

## Provenance

- Source of record: Kilo project memory (`project.md`), record `experimental_hierarchy`.
- Repo-side cross-references:
  - `training/results/EXPERIMENT_C_EXPLORATION_ABLATION_100k.md` — C report, recommends D.
  - `training/results/EXPERIMENT_D_QUALIFICATION.md` — canonical D report.
  - `training/results/DIAGNOSIS_100k_POST_STRIP_4SEED.md` — recommends C as next step.
