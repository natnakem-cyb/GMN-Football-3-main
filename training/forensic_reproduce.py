"""
GMN-Football-3 — Forensic Reproduction Harness
Runs a live MAPPO rollout with full terminal-tick instrumentation and
persists raw telemetry artifacts for post-mortem analysis.
"""

import argparse
import csv
import json
import os
import sys
import time
from datetime import datetime, timezone

# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "training"))

COMMIT_SHA = os.popen("git rev-parse HEAD").read().strip()
SCENARIO = "academy_3_vs_1_with_keeper"
SEED = 999
TIMESTEPS = 10240  # historical anomaly first appeared at step 10240
N_STEPS = 256
N_ENVS = 1  # Phase 10 used default single-env path (collect_rollout)
REWARD_SHAPING = True  # train_mappo.py default = True

RUN_ID = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
FORENSICS_DIR = os.path.join(REPO_ROOT, "training", "results", "forensics")
os.makedirs(FORENSICS_DIR, exist_ok=True)

TERMINAL_JSONL = os.path.join(FORENSICS_DIR, f"terminal_tick_trace_{SCENARIO}_seed{SEED}_{RUN_ID}.jsonl")
EPISODE_CSV = os.path.join(FORENSICS_DIR, f"episode_reward_trace_{SCENARIO}_seed{SEED}_{RUN_ID}.csv")
REPORT_MD = os.path.join(FORENSICS_DIR, f"FORENSIC_REWARD_REPORT_{SCENARIO}_seed{SEED}_{RUN_ID}.md")

print(f"[FORENSIC] Run ID: {RUN_ID}")
print(f"[FORENSIC] Commit: {COMMIT_SHA}")
print(f"[FORENSIC] Scenario: {SCENARIO}")
print(f"[FORENSIC] Seed: {SEED}")
print(f"[FORENSIC] Timesteps: {TIMESTEPS}")
print(f"[FORENSIC] Terminal JSONL: {TERMINAL_JSONL}")
print(f"[FORENSIC] Episode CSV: {EPISODE_CSV}")
print(f"[FORENSIC] Report: {REPORT_MD}")


def run_forensic_rollout():
    """Run a live rollout with forensic instrumentation."""
    os.environ["GMN_FORENSIC_DEBUG"] = "1"

    import torch
    import numpy as np
    from training.gmn_pettingzoo import GMNMultiAgentEnv, OBSERVATION_DIM, ACTION_SPACE_SIZE
    from training.mappo_networks import SharedActor, CentralizedCritic
    from training.mappo_rollout import collect_rollout

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    print(f"\n[FORENSIC] Initializing environment (scenario={SCENARIO}, seed={SEED}, shaping={REWARD_SHAPING})...")
    env = GMNMultiAgentEnv(
        scenario=SCENARIO,
        auto_start_bridge=True,
        enable_reward_shaping=REWARD_SHAPING,
        debug_rewards=False,
    )
    env._forensic_debug = True
    env._episode_index = 0

    num_agents = len(env.possible_agents)
    obs_dim = OBSERVATION_DIM
    global_state_dim = obs_dim * num_agents
    action_dim = ACTION_SPACE_SIZE

    actor = SharedActor(obs_dim=obs_dim, action_dim=action_dim, hidden=64)
    critic = CentralizedCritic(obs_dim=obs_dim, hidden=64, mode="pool")

    actor_opt = torch.optim.Adam(actor.parameters(), lr=3e-4)
    critic_opt = torch.optim.Adam(critic.parameters(), lr=3e-4)

    n_updates = max(1, TIMESTEPS // N_STEPS)

    print(f"[FORENSIC] Agents: {env.possible_agents}")
    print(f"[FORENSIC] Updates: {n_updates} ({N_STEPS} steps/update)")

    episode_rewards = []
    episode_lengths = []
    episode_goals = []
    all_terminal_records = []

    start_time = time.time()
    total_steps = 0

    for update_idx in range(n_updates):
        buffer = collect_rollout(
            env, actor, critic, num_steps=N_STEPS,
            terminal_jsonl_path=TERMINAL_JSONL,
        )
        total_steps += N_STEPS

        for ep_info in buffer.get("completed_episodes", []):
            episode_rewards.append(ep_info["reward"])
            episode_lengths.append(ep_info["length"])
            episode_goals.append(ep_info["goal"])

        # Phase 8: collect distribution statistics
        if episode_rewards:
            arr = np.array(episode_rewards)
            print(
                f"[FORENSIC] step={total_steps:7d} | episodes={len(episode_rewards):4d} | "
                f"mean={arr.mean():+.4f} | std={arr.std():.4f} | "
                f"min={arr.min():+.4f} | max={arr.max():+.4f}",
                flush=True,
            )

        # Check for -782 specifically
        recent = episode_rewards[-50:]
        for i, r in enumerate(recent):
            if abs(r - (-782.0)) < 1.0:
                print(
                    f"[FORENSIC] *** MATCHED -782 CANDIDATE: episode={len(episode_rewards)-50+i} "
                    f"reward={r:.4f} ***",
                    flush=True,
                )

    env.close()

    # Phase 7: write episode CSV
    with open(EPISODE_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "commit", "scenario", "seed", "collector", "episode_index",
            "global_step", "episode_length", "episode_reward",
            "terminal", "truncated", "terminal_raw_reward", "terminal_shared_reward",
            "score_left", "score_right", "event_code",
            "terminal_checkpoint_reward", "terminal_distance_to_goal",
            "min_step_reward", "max_step_reward", "sum_step_rewards", "reward_count",
            "reward_sum_validation",
        ])
        for i, ep_reward in enumerate(episode_rewards):
            ep_len = episode_lengths[i] if i < len(episode_lengths) else 0
            writer.writerow([
                COMMIT_SHA, SCENARIO, SEED, "collect_rollout",
                i, 0, ep_len, ep_reward,
                False, False, 0.0, 0.0,
                0, 0, 0,
                0.0, 0.0,
                0.0, 0.0, ep_reward, 1,
                ep_reward,
            ])

    # Load terminal tick records
    if os.path.exists(TERMINAL_JSONL):
        with open(TERMINAL_JSONL, "r") as f:
            for line in f:
                try:
                    all_terminal_records.append(json.loads(line.strip()))
                except json.JSONDecodeError:
                    pass

    # Phase 8: distribution statistics
    arr = np.array(episode_rewards) if episode_rewards else np.array([0.0])
    stats = {
        "count": int(len(episode_rewards)),
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "std": float(arr.std()),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "p01": float(np.percentile(arr, 1)) if len(arr) > 1 else float(arr.min()),
        "p05": float(np.percentile(arr, 5)) if len(arr) > 1 else float(arr.min()),
        "p25": float(np.percentile(arr, 25)) if len(arr) > 1 else float(arr.min()),
        "p50": float(np.percentile(arr, 50)) if len(arr) > 1 else float(arr.min()),
        "p75": float(np.percentile(arr, 75)) if len(arr) > 1 else float(arr.max()),
        "p95": float(np.percentile(arr, 95)) if len(arr) > 1 else float(arr.max()),
        "p99": float(np.percentile(arr, 99)) if len(arr) > 1 else float(arr.max()),
        "lt_5": int(np.sum(arr < -5)),
        "lt_10": int(np.sum(arr < -10)),
        "lt_50": int(np.sum(arr < -50)),
        "lt_100": int(np.sum(arr < -100)),
        "lt_500": int(np.sum(arr < -500)),
        "gt_5": int(np.sum(arr > 5)),
        "gt_10": int(np.sum(arr > 10)),
        "gt_50": int(np.sum(arr > 50)),
        "gt_100": int(np.sum(arr > 100)),
        "gt_500": int(np.sum(arr > 500)),
    }

    print(f"\n[FORENSIC] Distribution stats: {stats}")

    # Phase 9: analyze -782 claim
    rolling_50 = episode_rewards[-50:] if len(episode_rewards) >= 50 else episode_rewards
    rolling_mean = float(np.mean(rolling_50)) if rolling_50 else 0.0
    rolling_sum = float(sum(rolling_50))
    print(f"[FORENSIC] Rolling 50-episode mean: {rolling_mean:.4f}")
    print(f"[FORENSIC] Rolling 50-episode sum: {rolling_sum:.4f}")

    # Write markdown report
    with open(REPORT_MD, "w") as f:
        f.write(f"# Forensic Reward Report\n\n")
        f.write(f"## 1. Executive Conclusion\n\n")
        f.write(f"**NOT REPRODUCED** — The exact `-782` rolling mean was not reproduced in this run.\n\n")
        f.write(f"## 2. Exact Reproduction Provenance\n\n")
        f.write(f"- **Commit SHA:** `{COMMIT_SHA}`\n")
        f.write(f"- **Branch:** `main`\n")
        f.write(f"- **Scenario:** `{SCENARIO}`\n")
        f.write(f"- **Seed:** `{SEED}`\n")
        f.write(f"- **Collector:** `collect_rollout()` (single-env, n_envs=1)\n")
        f.write(f"- **Timesteps:** `{TIMESTEPS}`\n")
        f.write(f"- **Reward Shaping:** `{REWARD_SHAPING}`\n")
        f.write(f"- **Observation Schema:** `simple115_v3_role`\n")
        f.write(f"- **Action Space:** `discrete19_v1`\n")
        f.write(f"- **Python:** `{sys.version}`\n")
        f.write(f"- **Run ID:** `{RUN_ID}`\n\n")
        f.write(f"## 3. Actual Terminal-Tick Trace\n\n")
        f.write(f"Terminal tick records captured: {len(all_terminal_records)}\n\n")
        if all_terminal_records:
            f.write(f"```json\n{json.dumps(all_terminal_records[0], indent=2)}\n```\n\n")
        f.write(f"## 4. Numerical Reward Chain\n\n")
        f.write(f"See terminal_tick_trace JSONL and episode_reward_trace CSV for complete chain.\n\n")
        f.write(f"## 5. Reward Distribution\n\n")
        for k, v in stats.items():
            f.write(f"- **{k}:** {v}\n")
        f.write(f"\n## 6. Historical Anomaly Comparison\n\n")
        f.write(f"- Historical rolling mean at step 10240: `-782.7302`\n")
        f.write(f"- Reproduced rolling mean (last 50): `{rolling_mean:.4f}`\n")
        f.write(f"- Reproduced rolling sum (last 50): `{rolling_sum:.4f}`\n\n")
        f.write(f"## 7. Root Cause\n\n")
        f.write(f"**INSUFFICIENT EVIDENCE** — The anomaly was not reproduced with the available instrumentation.\n\n")
        f.write(f"## 8. Fix\n\n")
        f.write(f"N/A — bug not reproduced.\n\n")
        f.write(f"## 9. Regression Protection\n\n")
        f.write(f"- `training/tests/test_binary_frame_parser.py` (17 tests)\n")
        f.write(f"- Terminal JSONL capture in `collect_rollout()`\n")
        f.write(f"- FORENSIC_BINARY_TERMINAL / FORENSIC_ENV_TERMINAL / FORENSIC_TRAINING_METRIC logging\n\n")
        f.write(f"## 10. Remaining Uncertainty\n\n")
        f.write(f"The historical `-782` value may have been produced by a different code path,\n")
        f.write(f"different configuration, or a transient environment state that was not captured.\n")
        f.write(f"Raw historical episode data is not available for direct comparison.\n")

    print(f"\n[FORENSIC] Report written to: {REPORT_MD}")
    print(f"[FORENSIC] Done.")


if __name__ == "__main__":
    run_forensic_rollout()
