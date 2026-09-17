# Experimental Hierarchy

## Locked sequence

| Label | Name | Freeze / settings | Scope | Status |
|-------|------|-------------------|-------|--------|
| **A** | Frozen post-strip reference | Post-strip reward implementation, `E=1/β=0.03`, 100k timesteps | Reference baseline | Done |
| **B** | Horizon extension | Same freeze as A | 200k–500k timesteps | Conditionally unblocked after D exits; requires B brief documenting clampPassDirection scope and confirming no reward/strip edits |
| **C** | Controlled exploration ablation | Fixed horizon (100k), isolate exploration bonus `E` | Compare `(E=0, β=0)` vs `(E=1, β=0.03)` | Done |
| **D** | Tackle exploitation forensic investigation | No training; probe-only analysis on frozen engine | Attribute tackle-related rewards/outcomes at episode and transition level; identify causal mechanism before any intervention | In progress |

## Dependencies and gating

- **A** is the locked reference. Do not retrain A.
- **B** must not start until the primary case is qualified and the structural issues identified in C/D are resolved.
- **C** was the characterization step for the exploration axis. Its finding was that exploration at `(E=1, β=0.03)` does not improve productive football behavior relative to `(E=0, β=0)` at 100k; the dominant pathologies are structural (masking, possession collapse, reward attribution), not horizon-related.
- **D** is the tackle exploitation forensic investigation. Recurrence of pathological tackle behavior across B and C (seeds 7 and 999 in B; seed 123 in C) elevates it to a repeated behavioral failure signature. The exit criterion is identification of the causal mechanism that makes tackle-related behavior attractive despite zero scoring. Only after D establishes the mechanism may a single targeted intervention be designed. B is gated on D's forensic exit, not on time.

## Provenance

- Source of record: Kilo project memory (`project.md`), record `experimental_hierarchy`.
- Repo-side cross-references:
  - `training/results/EXPERIMENT_C_EXPLORATION_ABLATION_100k.md` — C report, recommends D.
  - `training/results/EXPERIMENT_D_QUALIFICATION.md` — canonical D report.
  - `training/results/DIAGNOSIS_100k_POST_STRIP_4SEED.md` — recommends C as next step.
  - `training/results/EXPERIMENT_B_PROVENANCE.md` — B closed as negative result; 0.0% goal rate all seeds; tackle spam identified.
  - `training/results/EXPERIMENT_B_HORIZON_200k.md` — B best-vs-final gap investigation; manifest `best_deterministic_goal_rate` values excluded as non-reproducible.
  - `training/results/D_EXPERIMENT_TACKLE_FORENSICS.md` — D tackle exploitation forensic investigation plan.
