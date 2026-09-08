# Phase 15 — RL Maturity Update

## Current Assessment: Level B → Level C (Pending Verification)

### Level B Criteria (Current)
- [x] Infrastructure works (bridge, environment, training loop)
- [x] Learning occurs (reward increases over training)
- [x] Basic evaluation pipeline functional
- [x] Per-agent reward pipeline verified

### Level C Criteria (Behavioral Learning)
Required:
- [ ] Policies demonstrate meaningful task behavior (passing, shooting, scoring)
- [ ] Behavior is reproducible across seeds (CV < 0.3)
- [ ] Reward redesign produces intended behavioral changes
- [ ] Regression tests prevent reward exploitation

### Current Status

**NOT YET VERIFIED** — The current checkpoints (pre-fix) show:
- Pass accuracy: 0%
- Shots per episode: 0.00
- Completed passes per episode: 0.00–0.44

These metrics indicate the policies have NOT learned meaningful football behavior. The reward redesign and bug fixes are necessary but not sufficient — fresh training is required to verify Level C.

### Promotion Path

1. Complete Phase 10 retraining (3 seeds)
2. Evaluate behavioral metrics (Phase 11)
3. Verify multi-seed stability (Phase 13)
4. Test generalization (Phase 14)
5. If all criteria met: promote to Level C
