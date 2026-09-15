# Brief: Real Retrain From Scratch (Parts 1–6 Done)

## Part A — ONNX Opponent Test Investigation

**Goal:** Root-cause `test_onnx_opponent.py::test_learned_opponent_snapshot` failure: checksums identical between snapshot ONNX policy and rule-based opponent, meaning the ONNX path is silently falling back to rule-based play.

**Investigation steps:**
1. Read `training/opponent_pool.py` to find `apply()` and `sample()` — confirm whether snapshot entries actually invoke ONNX inference or fall through to rule-based.
2. Read `training/bridge_server.ts` ONNX inference path (`runOnnxInference`, `getSnapshotSession`) — check whether the ONNX session is actually being created and called, or whether failures are swallowed silently.
3. Read `training/tests/test_onnx_opponent.py` to confirm what the test actually asserts and whether the checksum comparison is valid.
4. Run `python -m pytest training/tests/test_onnx_opponent.py -q` to get the actual current failure.

**Decision point:**
- If root cause is a clear, narrow fix within this brief's scope (e.g. session not being passed to `apply()`, wrong method call), fix it in `opponent_pool.py` or `bridge_server.ts` and re-run the test.
- If root cause is a larger ONNX export/inference mismatch unrelated to masking (e.g. exported weights don't match runtime inference, ort session creation failure), write one paragraph explaining it and leave the test failing. Do not expand scope.

**Commit:** If fixed, commit as `Part A: fix ONNX opponent snapshot fallback to rule-based`.

---

## Part B — Real Retrain From Scratch

**Preconditions:**
- Working tree clean (except Part A commit if applicable).
- HEAD is `24ee543` or later.
- No prior checkpoints are reused.

**Training steps (4 seeds, sequential):**

For each seed in `42, 123, 999, 7`:
1. Create a fresh output directory: `training/models/retrain_maskfix_seed{seed}/`
2. Run:
   ```
   python -m training.train_mappo --scenario academy_3_vs_1_with_keeper --timesteps 200000 --seed {seed} --models-dir training/models/retrain_maskfix_seed{seed}
   ```
   - `train_mappo.py` defaults: `training_mode=True`, `shot_clock_truncates=False`, `shot_clock_t_max=600` (lines 185–186, 199–203). No `--training-mode` flag needed; the default is already True for training.
3. Capture wall time from the training log (last few lines show elapsed time if logged; otherwise use the log file timestamps).
4. After training completes, compute SHA-256 of the `*_clean.pt` checkpoint.

**Eval steps (per seed):**
```
python training/eval_mappo_comprehensive.py --checkpoint-path training/models/retrain_maskfix_seed{seed}/mappo_academy_3_vs_1_with_keeper_seed{seed}_clean.pt --scenario academy_3_vs_1_with_keeper --num-episodes 50 --deterministic --force
```
- Confirm `training_mode` is NOT passed to eval (the eval script does not accept it; `GMNMultiAgentEnv` defaults to `training_mode=False`).
- Capture: goal rate, shots/ep, pass completion, shot-saved/blocked breakdown, tackles/ep.

**Artifacts to produce:**
- 4 fresh checkpoints in `training/models/retrain_maskfix_seed{seed}/`
- 4 eval JSON outputs
- `training/results/BASELINE_200k_MASKING_FIXED_RETRAIN.md` with:
  - Per-seed provenance table: exact command, wall time, checkpoint SHA-256
  - Eval results table: goal rate, shots/ep, pass completion, shot-saved/blocked, tackles/ep
  - Three-way comparison: `a4c16c0` vs stale `e2509b0`-on-new-eval vs this retrain
  - Direct verdict: did shooting/scoring behavior change?

**Commit:** `Part B: real 4-seed retrain from scratch with mask fixes + baseline doc`

---

## Out-of-Scope Guardrails
- Do not modify `reward_adapters.py`, `mappo_networks.py`, `mappo_update.py`, or any engine physics.
- Do not reuse, move, or relabel any pre-existing checkpoint.
- If training surfaces an unexpected failure, stop and report it — do not patch inline.
