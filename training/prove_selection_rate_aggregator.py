"""
Prove and verify the selection-rate aggregator on a single checkpoint.
"""
import os
import sys
import torch
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from training.gmn_pettingzoo import GMNMultiAgentEnv
from training.mappo_networks import SharedActor
from training.mappo_rollout import unwrap_obs, unwrap_masks, _mask_matrix

ACTION_NAMES = [
    "IDLE", "LEFT", "RIGHT", "UP", "DOWN",
    "UP_LEFT", "UP_RIGHT", "DOWN_LEFT", "DOWN_RIGHT",
    "SHORT_PASS", "LONG_PASS", "HIGH_PASS",
    "SHOT", "SPRINT",
    "RELEASE_DIRECTION", "RELEASE_SPRINT",
    "SLIDING", "DRIBBLE", "RELEASE_DRIBBLE"
]
PASS_ACTIONS = {9, 10, 11}
SHOT_ACTION = 12
PASS_SHOT_ACTIONS = {9, 10, 11, 12}

# ==============================================================================
# Task 1 — Freeze one definition
# selected_pass_shot_rate = count(action_selected ∈ {9,10,11,12} AND on_ball) / count(on_ball)
# Evaluation is deterministic=True (argmax policy).
# If stochastic, re-running would give different numbers.
# ==============================================================================
def compute_selected_pass_shot_rate(trace_rows):
    """
    selected_pass_shot_rate = count(action_selected ∈ {9,10,11,12} AND on_ball) / count(on_ball)

    Evaluation mode: deterministic=True (argmax policy over actor logits).
    Note: If stochastic (sampling actions from dist.sample()), re-running would give
    different numbers across runs due to random sampling.
    """
    count_onball = 0
    count_selected_pass_shot_onball = 0
    for r in trace_rows:
        if r["on_ball"]:
            count_onball += 1
            if r["selected_action"] in PASS_SHOT_ACTIONS:
                count_selected_pass_shot_onball += 1

    rate = count_selected_pass_shot_onball / count_onball if count_onball > 0 else 0.0
    return count_onball, count_selected_pass_shot_onball, rate


# ==============================================================================
# Task 2 — pi (probability-mass) path
# mean_pi_pass_shot = mean over on-ball rows of sum_{a in {9,10,11,12}} dist.probs[a]
# Source of truth: dist.probs == softmax over the actor's MASKED logits
# (SharedActor.forward sets illegal logits to -inf before Categorical(logits=...)).
# It is NOT derived from any selection counter. Same single loop as the trace rows.
# ==============================================================================
def compute_mean_pi_pass_shot(trace_rows):
    """mean_pi_pass_shot = mean(sum_{a in {9,10,11,12}} pi[a]) over on-ball rows."""
    vals = [r["pi_pass_shot"] for r in trace_rows if r["on_ball"]]
    return (sum(vals) / len(vals)) if vals else 0.0


def run_single_episode_trace(
    checkpoint_path: str = "training/models/mappo_academy_3_vs_1_with_keeper_seed999_100352.pt",
    scenario: str = "academy_3_vs_1_with_keeper",
    ep_seed: int = 502018,
    bridge_port: int = 5050,
):
    """
    Task 2 & 3: Single source of truth.
    One loop produces both the printed trace rows and the counter feeding the rate.
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    ckpt = torch.load(checkpoint_path, map_location="cpu")
    obs_dim = ckpt.get("obs_dim", 127)
    action_dim = ckpt.get("action_dim", 19)
    actor = SharedActor(obs_dim=obs_dim, action_dim=action_dim, hidden=64)
    actor.load_state_dict(ckpt["actor"])
    actor.eval()

    env = GMNMultiAgentEnv(scenario=scenario, auto_start_bridge=True, port=bridge_port)
    obs_dict, _ = env.reset(seed=ep_seed)
    controllable_agents = list(env.possible_agents)

    trace_rows = []
    all_decisions = []
    step_idx = 0

    try:
        while True:
            current_masks = unwrap_masks(obs_dict)
            obs_dict = unwrap_obs(obs_dict)
            current_agents = list(env.agents if env.agents else controllable_agents)
            pre_step_ball_owner = getattr(env, "_last_ball_owner_agent_idx", 255)

            local_obs = np.stack([obs_dict[a] for a in current_agents], axis=0).astype(np.float32)
            mask_matrix = _mask_matrix(current_masks, current_agents)

            with torch.no_grad():
                obs_tensor = torch.from_numpy(local_obs).float()
                mask_tensor = torch.tensor(mask_matrix, dtype=torch.bool)
                dist = actor(obs_tensor, mask_tensor)
                # Deterministic evaluation = True (argmax)
                actions = dist.logits.argmax(dim=-1)
                # Task 2: pi mass straight from softmax over the actor's masked logits.
                probs = dist.probs

            action_dict = {}
            for i, a in enumerate(current_agents):
                act_int = int(actions[i].item())
                action_dict[a] = act_int
                is_on_ball = (pre_step_ball_owner == i)

                agent_probs = probs[i].cpu().numpy()
                pi_pass_shot = float(sum(agent_probs[ac] for ac in sorted(PASS_SHOT_ACTIONS)))

                row_obj = {
                    "tick": step_idx,
                    "agent": a,
                    "agent_idx": i,
                    "on_ball": is_on_ball,
                    "selected_action": act_int,
                    "selected_action_name": ACTION_NAMES[act_int],
                    "pi_pass_shot": pi_pass_shot,
                    "pass_legal": any(mask_matrix[i][p] for p in PASS_ACTIONS),
                    "shot_legal": bool(mask_matrix[i][SHOT_ACTION]),
                }
                all_decisions.append(row_obj)
                if is_on_ball:
                    trace_rows.append(row_obj)

            obs_dict, rewards, terms, truncs, infos = env.step(action_dict)
            step_idx += 1
            term = any(terms.values()) if terms else False
            trunc = any(truncs.values()) if truncs else False
            if term or trunc or not env.agents:
                break
    finally:
        env.close()

    return trace_rows, all_decisions


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="training/models/mappo_academy_3_vs_1_with_keeper_seed999_100352.pt")
    ap.add_argument("--ep-seed", type=int, default=502018)  # Episode 3 of base_seed=500000
    ap.add_argument("--bridge-port", type=int, default=5050)
    args = ap.parse_args()

    checkpoint_path = args.checkpoint
    ep_seed = args.ep_seed

    import hashlib
    sha = hashlib.sha256(open(checkpoint_path, "rb").read()).hexdigest()
    ck = torch.load(checkpoint_path, map_location="cpu")

    print("=" * 80)
    print("PROVE AND VERIFY SELECTION-RATE AGGREGATOR")
    print(f"Checkpoint : {checkpoint_path}")
    print(f"SHA256     : {sha}")
    print(f"Timesteps  : {ck.get('timesteps')}")
    print(f"Episode    : ep_seed={ep_seed} (base_seed=500000 convention: ep1=500000, ep2=501009, ep3=502018)")
    print("Evaluation : deterministic=True (argmax policy)")
    print("=" * 80)

    trace_rows, all_decisions = run_single_episode_trace(
        checkpoint_path=checkpoint_path,
        ep_seed=ep_seed,
        bridge_port=args.bridge_port,
    )

    # Print trace table
    print("\nPER-TICK TRACE TABLE (All On-Ball Ticks):")
    print("-" * 80)
    print(f"| {'#':>3} | {'Tick':>4} | {'Agent':>6} | {'On-Ball':>7} | {'Action':>6} | {'Action Name':<17} | {'pi_pass_shot':>12} | {'Pass Legal':>10} | {'Shot Legal':>10} |")
    print("|" + "-"*5 + "|" + "-"*6 + "|" + "-"*8 + "|" + "-"*9 + "|" + "-"*8 + "|" + "-"*19 + "|" + "-"*14 + "|" + "-"*12 + "|" + "-"*12 + "|")
    for idx, r in enumerate(trace_rows):
        print(f"| {idx+1:>3} | {r['tick']:>4} | {r['agent']:>6} | {str(r['on_ball']):>7} | {r['selected_action']:>6} | {r['selected_action_name']:<17} | {r['pi_pass_shot']:>12.6f} | {str(r['pass_legal']):>10} | {str(r['shot_legal']):>10} |")
    print("-" * 80)

    # Aggregator output from the EXACT same trace_rows
    count_onball, count_selected_pass_shot_onball, rate = compute_selected_pass_shot_rate(trace_rows)
    mean_pi_ps = compute_mean_pi_pass_shot(trace_rows)

    print("\nAGGREGATOR OUTPUT (Mechanically derived from above trace table):")
    print(f"count_onball                     = {count_onball}")
    print(f"count_selected_pass_shot_onball = {count_selected_pass_shot_onball}")
    print(f"selected_pass_shot_rate         = {rate:.6f} ({rate*100:.2f}%)")
    print(f"mean_pi_pass_shot                = {mean_pi_ps:.6f} ({mean_pi_ps*100:.4f}%)")

    # Manual verification by code check
    manual_count = sum(1 for r in trace_rows if r["selected_action"] in {9, 10, 11, 12})
    print(f"\nHand-count of rows where selected in {{9,10,11,12}}: {manual_count}")
    print(f"Does hand-count equal count_selected_pass_shot_onball? {manual_count == count_selected_pass_shot_onball}")

    # Whole-run reconciliation
    N_all_decisions = len(all_decisions)
    N_pass_shot_selected = sum(1 for r in all_decisions if r["selected_action"] in {9, 10, 11, 12})
    whole_run_rate = N_pass_shot_selected / N_all_decisions if N_all_decisions > 0 else 0.0

    print("\nWHOLE-RUN ACTION-FREQUENCY (Same Run Data):")
    print(f"N_all_decisions          = {N_all_decisions}")
    print(f"N_pass_shot_selected     = {N_pass_shot_selected}")
    print(f"N_pass_shot_selected / N_all_decisions = {N_pass_shot_selected} / {N_all_decisions} = {whole_run_rate:.6f} ({whole_run_rate*100:.4f}%)")
    print("=" * 80)


if __name__ == "__main__":
    main()

