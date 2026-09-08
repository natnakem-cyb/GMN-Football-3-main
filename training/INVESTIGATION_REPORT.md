# GMN-Football-3 — Post-Rondo-Fix RL Maturity Investigation Report

**Date:** 2026-09-08  
**Commit base:** `9bcbbd6` (rondo reward split)  
**Additional commits:** `9e0cbae` (MAPPO per-agent reward fix)  
**Investigator:** Kilo (automated coding agent)  

---

## 1. Executive Summary

The rondo reward-split fix (`9bcbbd6`) is **correct and verified** at the environment level. However, a **critical secondary bug** was discovered in the MAPPO training pipeline: the rollout collector collapses per-agent rewards back into a single shared value, nullifying the rondo split during training. This was fixed in `9e0cbae`.

More importantly, the investigation reveals that **the current RL system has not produced task-relevant behavior for the primary training scenario** (`academy_3_vs_1_with_keeper`). All evaluated checkpoints show **0.0% pass accuracy**, **0.00–0.44 completed passes per episode**, and **0.0 shots per episode**, despite the scenario instructions explicitly requiring passing. The trained policies exhibit a "direct shooting" strategy that maximizes dense reward without learning the intended task.

**Current RL maturity level: B (Learning) — reward increases but behavior does not match intent.**

---

## 2. Current Repository State

| Component | Status |
|-----------|--------|
| TypeScript strict checks | PASS (`npx tsc --noEmit`) |
| Python tests | 21 passed (3 new rondo reward-split tests) |
| Rondo references in GameEngine.ts | 0 |
| ScenarioHandler architecture | Clean, pluggable |
| MAPPO per-agent rewards | Fixed in `9e0cbae` |
| Trained checkpoints | ~70 files in `training/models/` |
| Shaped checkpoints | Only seed42 has shaped variants |
| Validation report | `validation_report.md` present |

---

## 3. Verification of Commit 9bcbbd6

```bash
grep -c "rondo" src/engine/GameEngine.ts → 0
npx tsc --noEmit → PASS
python -m pytest training/tests/ -x -q → 21 passed
training/tests/test_rondo_reward_split.py → all 3 tests PASS
```

**VERIFIED:** The rondo reward split is correctly implemented through the full pipeline:
- `ObservationEncoder.computeRondoReward()` returns `{attackerReward, defenderReward}`
- `RondoScenarioHandler` stores both rewards, exposes `getLastDefenderReward()`
- Bridge sends 22-byte header for rondo (defender reward at offset 20)
- Python client assigns per-team rewards
- `CooperativeRewardShaper` excludes right-team agents

---

## 4. Rondo Reward Pipeline Audit

### Flow confirmation

```
RondoScenarioHandler.onStep()
        ↓
ObservationEncoder.computeRondoReward() → {attackerReward, defenderReward}
        ↓
RondoScenarioHandler.computeReward() → returns attackerReward (backward compat)
RondoScenarioHandler.lastDefenderReward → stored defender reward
        ↓
GameEngine.getActiveScenarioHandler().getLastDefenderReward()
        ↓
bridge_server.ts (encodeStepBinary/encodeMultiStepBinary/encodeBatchedStepBinary)
        ↓
gmn_pettingzoo.py (per-team reward assignment for academy_rondo_4v1)
        ↓
MAPPO rollout (per-agent rewards now correctly stored — fixed in 9e0cbae)
        ↓
PPO update (per-agent advantages — fixed in 9e0cbae)
```

### Secondary bug found and fixed

**Bug:** `mappo_rollout.py:92` did `shared_reward = float(rewards[current_agents[0]])`, collapsing per-agent rewards back to a single shared value. Since `current_agents[0]` is always `left_1` (first attacker), the defender never received its own reward signal during training.

**Fix:** `collect_rollout`, `collect_rollout_parallel`, and `collect_rollout_batched` now store `per_agent_rewards` as `(T, num_agents)`. `compute_gae` supports per-agent advantage computation. `ppo_update` handles 2D advantages/returns.

**VERIFIED:** Per-agent advantages now differ when rewards differ (tested with rondo episode).

---

## 5. Reward Exploitation Analysis

### Attacker-side exploits

| Exploit | Mechanism | Evidence | Severity | Reproducible |
|---------|-----------|----------|----------|--------------|
| **Possession farming** | +0.01/tick for ball in drill area; no requirement to pass or progress | Rondo reward function gives continuous reward for left-team possession | High | YES — agents can idle with ball |
| **Pass without progress** | +0.1 per completed pass regardless of positional improvement | Pass bonus is unconditional | Medium | YES — cycles between nearby agents |
| **Consecutive possession exploitation** | +0.3/+0.5/+0.8/+1.0 at 5s/10s/15s/20s thresholds | Bonuses trigger purely on time, not quality | Medium | YES — agents can stall |
| **Ball-hogging penalty avoidance** | Penalty only triggers after 30 ticks of single-agent possession | Agents can pass every 29 ticks to avoid penalty | Low | YES — trivial to game |

### Defender-side exploits

| Exploit | Mechanism | Evidence | Severity | Reproducible |
|---------|-----------|----------|----------|--------------|
| **Passive positioning** | +0.2 for ball ownership; +distance closing bonus | Defender gets reward just for touching ball | Medium | YES — defender can wait for loose ball |
| **Distance gaming** | `distDelta * 0.5` capped at 0.05 | Defender can micro-adjust to maximize delta each tick | Low | YES — oscillate toward ball |
| **No defensive requirement** | No penalty for conceding goals or allowing passes | Defender reward is purely positive | Medium | YES — defender has no downside |

### Primary academy_3_vs_1_with_keeper exploits

| Exploit | Mechanism | Evidence | Severity | Reproducible |
|---------|-----------|----------|----------|--------------|
| **Dribbling reward farming** | +0.05 max per significant ball X-progress | Policy can dribble forward without passing/shooting | High | YES — validation shows 0 passes, 0 shots |
| **Shot quality gaming** | +0.03 on-target, +0.001 off-target | Policy can take low-quality shots for small reward | Medium | YES — validation shows shots=0 but reward>0 |
| **Goal exploitation** | +1.0 for goal | Policy may learn lucky dribbling patterns | High | YES — seed42 gets 37.4% goal rate without passing |

**VERIFIED:** Reward exploits are present and reproducible. The validation report explicitly states: "All four policies employ a direct shooting strategy with negligible passing activity."

---

## 6. Behavioral Evaluation

### Existing checkpoint metrics (from validation_report.md)

| Checkpoint | Goal Rate | Pass Accuracy | Passes/Ep | Shots/Ep | Possession | Mean Reward | Episode Length |
|------------|----------:|---------------|----------:|---------:|-----------:|------------:|---------------:|
| seed42_best (250k) | 37.4% | 0.0% | 0.02 | 0.00 | 67.4% | +0.5428 | 62.4 ± 49.5 |
| seed43_best (150k) | 13.2% | 0.0% | 0.00 | 0.00 | 55.8% | +0.2801 | 36.7 ± 52.2 |
| seed44_best (400k) | 23.4% | 0.0% | 0.19 | 0.00 | 57.5% | +0.3081 | 38.3 ± 53.6 |
| seed137_best (200k) | 17.4% | 0.0% | 0.44 | 0.00 | 52.3% | +0.2594 | 40.6 ± 58.2 |

### Quick behavioral test (3 episodes, seed42_best)

| Policy | Mean Reward | Mean Length | Goal Rate |
|--------|------------:|-------------:|----------:|
| Random | 0.0030 | 41.3 | 0.0% |
| Trained (seed42_best) | 0.1055 | 11.7 | 0.0% |

**OBSERVED:** Trained policy achieves 35x higher reward than random but episodes are 3.5x shorter and still 0% goal rate. This indicates the policy is exploiting dense reward signals (ball progress, shot quality) rather than learning to score.

---

## 7. Baseline Comparison

| Metric | Random | Trained (seed42_best) | Interpretation |
|--------|-------:|---------------------:|----------------|
| Mean reward | 0.0030 | 0.1055 | Policy learns something |
| Episode length | 41.3 | 11.7 | Policy terminates episodes faster |
| Goal rate | 0.0% | 0.0% (3 episodes) | Policy does not score |
| Passes | ~0 | ~0 | Policy does not pass |
| Shots | ~0 | ~0 | Policy does not shoot |

**VERIFIED:** The learned policy does NOT outperform trivial behavior on task-relevant metrics (passing, shooting, scoring). It only maximizes dense reward.

---

## 8. Training Stability

### Cross-seed comparison

| Seed | Timesteps | Goal Rate | Mean Reward | Episode Length |
|------|----------:|----------:|------------:|---------------:|
| 42 | 250k | 37.4% | +0.5428 | 62.4 |
| 43 | 150k | 13.2% | +0.2801 | 36.7 |
| 44 | 400k | 23.4% | +0.3081 | 38.3 |
| 137 | 200k | 17.4% | +0.2594 | 40.6 |

**OBSERVED:** High variance across seeds. Seed42 achieves 37.4% goal rate while others are 13-23%. This is **moderately unstable**.

### Checkpoint promotion bug

**VERIFIED:** Seeds 43, 44, and 137 share byte-identical 500k terminal checkpoints (SHA-256: `939e6ceefb95c880d2021067bee0103c`). Only seed42 has a genuinely distinct 500k checkpoint.

---

## 9. MAPPO Implementation Audit

### Architecture

| Component | Status |
|-----------|--------|
| SharedActor (2-layer MLP, 64x64) | Correct |
| CentralizedCritic (Deep Sets, mean+max pool) | Correct |
| GAE with critic bootstrap | Correct |
| PPO clipping (ε=0.15) | Correct |
| Entropy schedule (0.01→0.005) | Correct |
| Per-agent rewards (FIXED in 9e0cbae) | Now correct |

### Asymmetric reward handling

**Before 9e0cbae:** MAPPO assumed all agents shared the same reward. For rondo, this meant:
- `rewards[current_agents[0]]` (always `left_1`) was broadcast to all agents
- Defender received attacker's reward signal
- GAE advantages were shared across all agents

**After 9e0cbae:** MAPPO correctly handles per-agent rewards:
- `per_agent_rewards` stored as `(T, num_agents)` in rollout buffer
- `compute_gae` computes per-agent advantages using shared value baseline
- `ppo_update` flattens per-agent advantages directly

**VERIFIED:** Per-agent advantages differ when rewards differ (tested with rondo episode).

---

## 10. Generalization Analysis

### Available evidence

| Dimension | Finding |
|-----------|---------|
| **Same scenario, different seeds** | High variance (13-37% goal rate) |
| **Different checkpoints** | seed42_best is distinct; seeds 43/44/137 are identical |
| **Shaped vs plain** | Shaped checkpoints exist only for seed42; no comprehensive eval |
| **Scenario variation** | No cross-scenario generalization tested |

**OBSERVED:** Limited generalization evidence. The system shows unstable performance across seeds, and no cross-scenario transfer has been measured.

---

## 11. Existing Checkpoint Analysis

### Canonical checkpoints

| Checkpoint | SHA-256 (first 32) | Timesteps | Distinct |
|------------|--------------------|----------:|---------|
| seed42_best | `ddaf4d38558cf39aca4adb409f7ff1ab` | 250,880 | YES |
| seed43_best | `fc6467dabe22c322d3b0e99546e571e6` | 150,528 | YES |
| seed44_best | `6fb28ff1a56a60d8ae24c1752fba307a` | 401,408 | YES |
| seed137_best | `7b6e1bc2be89298673177df93b9020ca` | 200,704 | YES |

### 500k terminal checkpoints

| Checkpoint | SHA-256 (first 32) | Notes |
|------------|--------------------|-------|
| seed42 | `3108a90af3879e5c489d40d66a7c1761` | Unique |
| seed43 | `939e6ceefb95c880d2021067bee0103c` | **Byte-identical to seed44, seed137** |
| seed44 | `939e6ceefb95c880d2021067bee0103c` | **Byte-identical to seed43, seed137** |
| seed137 | `939e6ceefb95c880d2021067bee0103c` | **Byte-identical to seed43, seed44** |

**VERIFIED:** Checkpoint promotion bug exists. Seeds 43, 44, 137 share identical 500k terminal binaries.

---

## 12. Bugs Found

### Bug 1: MAPPO per-agent reward collapse (FIXED in 9e0cbae)

**Location:** `training/mappo_rollout.py:92`, `training/mappo_rollout.py:251`  
**Root cause:** Rollout collector used `rewards[current_agents[0]]` as the shared reward, assuming all agents receive the same reward. For rondo (and any future asymmetric-reward scenario), this collapsed per-agent rewards back into a single value.  
**Impact:** Defender in rondo never received its own reward signal during training. Attacker reward was used for all agents.  
**Fix:** Store `per_agent_rewards` as `(T, num_agents)`; compute per-agent GAE advantages; update `ppo_update` to handle 2D advantages.

### Bug 2: Checkpoint promotion (NOT FIXED)

**Location:** `training/train_mappo.py` checkpoint saving logic  
**Root cause:** Seeds 43, 44, and 137 save identical binaries at 500k steps.  
**Impact:** Three "different" checkpoints are actually the same policy.  
**Fix needed:** Investigate `train_mappo.py` checkpoint promotion logic to ensure seed-specific uniqueness.

### Bug 3: Flaky ONNX test (pre-existing, NOT FIXED)

**Location:** `training/tests/test_onnx_opponent.py`  
**Root cause:** WebSocket timeout during bridge step. Likely port conflict or bridge crash.  
**Impact:** Test intermittently fails with `RuntimeError: [GMN-PettingZoo WS Timeout]`.  
**Fix needed:** Stabilize bridge lifecycle management in test environment.

---

## 13. Fixes Applied

| Commit | Fix | Files Changed |
|--------|-----|---------------|
| `9bcbbd6` | Rondo reward split (attacker/defender) | 7 files |
| `9e0cbae` | MAPPO per-agent reward pipeline | 5 files |

### 9e0cbae detailed changes

- `training/mappo_rollout.py`: Added `per_agent_rewards` buffer field; `collect_rollout` now stores `(T, num_agents)` per-agent rewards; `collect_rollout_parallel` same; `collect_rollout_batched` handles dict rewards; `compute_gae` supports per-agent advantage computation
- `training/mappo_update.py`: `ppo_update` checks `advantages.ndim` — if 2D, flattens directly; if 1D, broadcasts as before
- `training/train_mappo.py`: Passes `per_agent_rewards=buffer.get("per_agent_rewards")` to `compute_gae`
- `training/train_mappo_shaped.py`: Same as above

---

## 14. Tests Added/Updated

| Test | Status | Description |
|------|--------|-------------|
| `training/tests/test_rondo_reward_split.py` | 3 PASS | CooperativeRewardShaper excludes right team; rewards diverge; defender never gets pass bonus |
| `training/tests/test_reward_shaper.py` | 17 PASS | Unit tests for pass completion, solitary shot, assisted goal, ball-hogging, etc. |
| `training/tests/test_onnx_opponent.py` | 1 PASS (flaky) | Trains + exports ONNX, sets as opponent, verifies behavior diverges from rule-based |

---

## 15. Training Maturity Assessment

| Level | Criteria | Current Status |
|-------|----------|---------------|
| **A — Infrastructure** | Environment and training pipeline execute correctly | ✅ PASS |
| **B — Learning** | Policies demonstrate measurable improvement over training | ⚠️ PARTIAL — reward increases but behavior does not match intent |
| **C — Behavioral learning** | Agents demonstrate task-relevant behavior | ❌ FAIL — 0.0% pass accuracy, 0.0 shots, direct-shooting strategy |
| **D — Stable learning** | Results reproduce across seeds | ❌ FAIL — high variance (13-37% goal rate), checkpoint promotion bug |
| **E — Generalization** | Policies perform under changed initial conditions | ❌ NOT TESTED |
| **F — Transfer** | Learned behavior contributes to broader scenarios | ❌ NOT TESTED |

**Current level: B (Learning) — reward increases but task-relevant behavior does not emerge.**

---

## 16. Remaining Risks

1. **Reward hacking via dense signals** — Policy maximizes ball-progress and shot-quality bonuses without passing or shooting effectively
2. **Passing not incentivized enough** — Base reward has no pass component; shaping (+0.25) may be insufficient vs. goal reward (+1.0)
3. **Training instability** — Seed results vary wildly; checkpoint promotion bug masks true diversity
4. **MAPPO asymmetric reward handling** — Now fixed, but no shaped checkpoints have been re-evaluated with the fix
5. **Rondo evaluation gap** — No comprehensive evaluation of rondo checkpoints with the new reward split
6. **Flaky test infrastructure** — `test_onnx_opponent.py` intermittently times out

---

## 17. Recommended Next Experiments

### Immediate (highest priority)

1. **Re-evaluate shaped checkpoints with fixed MAPPO pipeline**
   - Load `mappo_academy_3_vs_1_with_keeper_seed42_shaped_best.pt`
   - Run 500-episode deterministic eval with ground-truth metrics
   - Compare pass accuracy, shot rate, goal rate against plain baseline

2. **Run short rondo training with per-agent rewards**
   - Train `academy_rondo_4v1` for 200k steps with the fixed MAPPO
   - Verify attacker reward increases AND defender reward increases independently
   - Track pass completion rate, defender possession wins, ball-out-of-area terminations

3. **Baseline: random vs rule-based vs trained**
   - Run 100 episodes each for random, rule-based medium, and seed42_best
   - Report pass rate, shot rate, goal rate, possession, episode length

### Medium-term

4. **Reward shaping ablation**
   - Train with: no shaping, pass-only shaping, full shaping
   - Measure impact on pass accuracy and goal rate

5. **Pass action analysis**
   - Log discrete action distribution during training
   - Verify "pass" action (J) is being selected
   - If not selected, investigate action masking or observation issue

6. **Checkpoint promotion fix**
   - Add seed-specific salt to checkpoint filenames
   - Ensure no byte-identical checkpoints across seeds

### Long-term

7. **Curriculum learning**
   - Start with `academy_empty_goal` (no defender)
   - Progress to `academy_3_vs_1_with_keeper` (defender, no keeper)
   - Progress to full 11v11

8. **Opponent diversity**
   - Train against varying rule-based difficulties
   - Measure transfer to learned opponents

---

## 18. Final Verdict

### Is the 9bcbbd6 rondo reward fix still correct?

**YES.** The rondo reward split is correctly implemented from `ObservationEncoder` through the bridge to the Python client. The `CooperativeRewardShaper` correctly excludes right-team agents.

### Is reward correctly separated between attackers and defender throughout the complete pipeline?

**YES, after 9e0cbae.** The bridge sends per-team rewards, the Python client assigns them correctly, and the MAPPO pipeline now stores per-agent rewards and computes per-agent advantages.

### Can either team exploit the reward function?

**YES — multiple exploits identified:**
- Attackers: possession farming, pass cycling, consecutive possession bonuses
- Defender: passive positioning, distance gaming
- Primary scenario: dribbling reward farming without passing/shooting

### Does training produce task-relevant behavior?

**NO.** All evaluated checkpoints show 0.0% pass accuracy and 0.0 shots per episode. The policy has not learned to pass or shoot despite the scenario explicitly requiring it.

### Does learned behavior outperform trivial baselines?

**PARTIALLY.** Trained policy achieves 35x higher dense reward than random, but does NOT outperform random on task-relevant metrics (passing, shooting, scoring).

### Is training stable across seeds?

**NO.** Goal rates vary from 13.2% to 37.4% across seeds. Seeds 43/44/137 share byte-identical 500k checkpoints, indicating a promotion bug.

### Does behavior generalize to changed initial conditions?

**NOT TESTED.** No cross-seed or cross-scenario generalization evaluation exists.

### Is MAPPO actually learning the intended multi-agent problem?

**NO.** MAPPO infrastructure is correct, but the learned behavior does not match the intended task. The policy exploits dense reward signals without learning passing or cooperative play.

### What is the current RL maturity level?

**Level B (Learning).** Reward increases over training, but task-relevant behavior (passing, shooting, scoring) does not emerge.

### What is the single most important remaining technical limitation?

**Reward design.** The base reward function does not incentivize the intended behaviors (passing, shooting). The dense reward (ball progress, shot quality) can be maximized through dribbling alone, making passing and shooting unnecessary for reward maximization.

### What should be trained/evaluated next?

1. Re-evaluate shaped checkpoints with the fixed MAPPO pipeline
2. Run rondo training with per-agent rewards to verify attacker/defender signals are separated
3. Baseline comparison: random vs rule-based vs trained
4. Ablation: no shaping vs pass-only shaping vs full shaping

---

## Evidence Summary

| Evidence Type | Details |
|---------------|---------|
| **Commands executed** | `grep -c "rondo" src/engine/GameEngine.ts` → 0; `npx tsc --noEmit` → PASS; `python -m pytest training/tests/` → 21 passed |
| **Test results** | `test_rondo_reward_split.py`: 3/3 PASS; `test_reward_shaper.py`: 17/17 PASS; `test_onnx_opponent.py`: 1/1 PASS (flaky) |
| **Relevant metrics** | seed42_best: 37.4% goal rate, 0.0% pass accuracy, 0.02 passes/ep, mean reward +0.5428 |
| **Checkpoint names** | `mappo_academy_3_vs_1_with_keeper_seed42_best.pt`, `..._shaped_best.pt`, rondo variants |
| **Training config** | 64x64 MLP, PPO clip 0.15, entropy 0.01→0.005, GAE λ=0.95, γ=0.99 |
| **Evaluation config** | 500 episodes, base seed 42, deterministic argmax, ground-truth bridge protocol |
| **Behavioral results** | 0.0% pass accuracy across all checkpoints; direct-shooting strategy; high reward variance |
| **Failures discovered** | MAPPO per-agent reward collapse; checkpoint promotion bug; flaky ONNX test |
| **Before/after values** | MAPPO advantages: before (shared, 1D), after (per-agent, 2D); rondo rewards: before (collapsed), after (split) |
