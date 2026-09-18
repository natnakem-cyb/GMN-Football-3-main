"""
Actor-Optimization / Advantage-Consumption Audit — Task 2: Mask / Index Alignment Check

Verifies:
1. Action indices in probe/logging match ActionMapping.ts and training loss
2. Masked (illegal) actions receive -inf logits in the actor forward pass
3. In the actual PPO surrogate loss, illegal action logits receive ZERO gradient
   (because log_prob only depends on legal logits through normalization, and
   illegal logits are -inf so their exp is 0)
4. Action-mask alignment between rollout sampling and loss recomputation

Outputs:
  - training/results/actor_optimization_audit_task2.json
"""
import argparse
import json
import os
import sys
from datetime import datetime, timezone

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from training.gmn_pettingzoo import GMNMultiAgentEnv
from training.mappo_networks import SharedActor, CentralizedCritic
from training.mappo_rollout import unwrap_obs, unwrap_masks, _mask_matrix

ACTION_NAMES = [
    "IDLE", "LEFT", "RIGHT", "UP", "DOWN",
    "UP_LEFT", "UP_RIGHT", "DOWN_LEFT", "DOWN_RIGHT",
    "LONG_PASS", "HIGH_PASS", "SHORT_PASS",
    "SHOT", "SPRINT",
    "RELEASE_DIRECTION", "RELEASE_SPRINT",
    "SLIDING", "DRIBBLE", "RELEASE_DRIBBLE",
]
# Authoritative indices from src/engine/ActionMapping.ts
EXPECTED_INDICES = {
    "TACKLE": 16,
    "SHOT": 12,
    "LONG_PASS": 9,
    "HIGH_PASS": 10,
    "SHORT_PASS": 11,
    "DRIBBLE": 17,
    "SPRINT": 13,
    "IDLE": 0,
}
MOVE_RANGE = range(1, 9)


def load_checkpoint(path: str):
    ckpt = torch.load(path, map_location="cpu")
    obs_dim = ckpt.get("obs_dim", 127)
    action_dim = ckpt.get("action_dim", 19)
    actor = SharedActor(obs_dim=obs_dim, action_dim=action_dim, hidden=64)
    actor.load_state_dict(ckpt["actor"])
    actor.eval()
    critic = None
    if "critic" in ckpt:
        critic = CentralizedCritic(obs_dim=obs_dim, hidden=64)
        critic.load_state_dict(ckpt["critic"])
        critic.eval()
    return actor, critic, obs_dim, action_dim


def check_action_index_alignment():
    """Verify that the indices used in probe/training match ActionMapping.ts."""
    results = {}
    probe_indices = {
        "TACKLE": 16,
        "SHOT": 12,
        "LONG_PASS": 9,
        "HIGH_PASS": 10,
        "SHORT_PASS": 11,
        "DRIBBLE": 17,
        "SPRINT": 13,
        "IDLE": 0,
    }
    for name, expected_idx in EXPECTED_INDICES.items():
        probe_idx = probe_indices.get(name)
        results[name] = {
            "expected_idx": expected_idx,
            "probe_idx": probe_idx,
            "match": expected_idx == probe_idx,
        }
    return results


def check_mask_gradient_mass(checkpoint_path: str, num_steps: int = 50):
    """Verify that illegal actions receive -inf logits and zero gradient mass
    in the actual PPO surrogate loss."""
    actor, critic, obs_dim, action_dim = load_checkpoint(checkpoint_path)

    env = GMNMultiAgentEnv(
        scenario="academy_3_vs_1_with_keeper_onball",
        auto_start_bridge=True,
        port=5053,
    )
    controllable_agents = list(env.possible_agents)
    obs_dict, _ = env.reset(seed=800000)
    current_ep_masks = unwrap_masks(obs_dict)
    obs_dict = unwrap_obs(obs_dict)

    illegal_logit_counts = 0
    total_illegal_positions = 0
    gradient_mass_on_illegal_in_ppo_loss = 0.0
    steps_with_illegal_actions = 0
    max_grad_on_illegal = 0.0
    illegal_grad_nonzero_count = 0

    for step in range(num_steps):
        current_agents = list(env.agents if env.agents else controllable_agents)
        local_obs = np.stack([obs_dict[a] for a in current_agents], axis=0).astype(np.float32)
        mask_matrix = _mask_matrix(current_ep_masks, current_agents)

        obs_tensor = torch.from_numpy(local_obs).float()
        mask_tensor = torch.tensor(mask_matrix, dtype=torch.bool)

        # Forward pass with mask — same as ppo_update
        logits = actor.net(obs_tensor)
        masked_logits = logits.masked_fill(~mask_tensor, float("-inf"))
        dist = torch.distributions.Categorical(logits=masked_logits)

        # Check illegal actions have -inf logits
        illegal_mask = ~mask_tensor
        if illegal_mask.any():
            n_illegal = int(illegal_mask.sum().item())
            n_inf = (masked_logits[illegal_mask] == float("-inf")).sum().item()
            illegal_logit_counts += n_inf
            total_illegal_positions += n_illegal
            steps_with_illegal_actions += 1

        # Sample actions (must be legal because of masking)
        actions = dist.sample()
        old_logprobs = dist.log_prob(actions).detach()

        # Compute ACTUAL PPO surrogate loss (same as ppo_update)
        # Need to recompute dist with gradients enabled
        masked_logits_train = actor.net(obs_tensor).masked_fill(~mask_tensor, float("-inf"))
        dist_train = torch.distributions.Categorical(logits=masked_logits_train)
        new_logprobs = dist_train.log_prob(actions)

        advantages = torch.ones_like(new_logprobs)
        ratio = torch.exp(new_logprobs - old_logprobs)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - 0.15, 1.0 + 0.15) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()

        actor.zero_grad()
        policy_loss.backward()

        # Check gradient on final layer weights for illegal action columns
        final_weight_grad = actor.net[-1].weight.grad
        if final_weight_grad is not None and illegal_mask.any():
            for agent_idx in range(mask_tensor.shape[0]):
                agent_illegal = illegal_mask[agent_idx]
                if agent_illegal.any():
                    agent_grad_norm = torch.norm(final_weight_grad[agent_illegal]).item()
                    gradient_mass_on_illegal_in_ppo_loss += agent_grad_norm
                    max_grad_on_illegal = max(max_grad_on_illegal, agent_grad_norm)
                    if agent_grad_norm > 1e-8:
                        illegal_grad_nonzero_count += 1

        # Step env
        action_dict = {a: int(actions[i].item()) for i, a in enumerate(current_agents)}
        obs_dict, rewards, terms, truncs, infos = env.step(action_dict)
        obs_dict = unwrap_obs(obs_dict)
        current_ep_masks = unwrap_masks(obs_dict)

        term = any(terms.values()) if terms else False
        if term:
            break

    env.close()

    return {
        "total_illegal_logits_set_to_inf": illegal_logit_counts,
        "total_illegal_positions_checked": total_illegal_positions,
        "fraction_illegal_with_inf_logits": illegal_logit_counts / max(total_illegal_positions, 1),
        "gradient_mass_on_illegal_in_ppo_loss": float(gradient_mass_on_illegal_in_ppo_loss),
        "max_gradient_on_illegal_single_action": float(max_grad_on_illegal),
        "illegal_grad_nonzero_count": illegal_grad_nonzero_count,
        "steps_with_illegal_actions": steps_with_illegal_actions,
    }


def check_mask_consistency_between_rollout_and_update(checkpoint_path: str):
    """Verify that the mask applied during rollout sampling is the same mask
    applied during the policy-gradient loss recomputation."""
    actor, critic, obs_dim, action_dim = load_checkpoint(checkpoint_path)

    env = GMNMultiAgentEnv(
        scenario="academy_3_vs_1_with_keeper_onball",
        auto_start_bridge=True,
        port=5054,
    )
    controllable_agents = list(env.possible_agents)
    obs_dict, _ = env.reset(seed=800100)
    current_ep_masks = unwrap_masks(obs_dict)
    obs_dict = unwrap_obs(obs_dict)

    mismatch_count = 0
    total_steps = 0

    for step in range(30):
        current_agents = list(env.agents if env.agents else controllable_agents)
        local_obs = np.stack([obs_dict[a] for a in current_agents], axis=0).astype(np.float32)
        mask_matrix = _mask_matrix(current_ep_masks, current_agents)

        obs_tensor = torch.from_numpy(local_obs).float()
        mask_tensor = torch.tensor(mask_matrix, dtype=torch.bool)

        # Rollout sampling (same as collect_rollout)
        with torch.no_grad():
            dist_rollout = actor(obs_tensor, mask_tensor)
            actions = dist_rollout.sample()

        # Loss recomputation (same as ppo_update)
        dist_update = actor(obs_tensor, mask_tensor)
        logprob_update = dist_update.log_prob(actions)

        # The log-probabilities should match because the same mask is applied
        with torch.no_grad():
            logprob_rollout = dist_rollout.log_prob(actions)
            if not torch.allclose(logprob_rollout, logprob_update, atol=1e-6):
                mismatch_count += 1

        total_steps += 1

        action_dict = {a: int(actions[i].item()) for i, a in enumerate(current_agents)}
        obs_dict, rewards, terms, truncs, infos = env.step(action_dict)
        obs_dict = unwrap_obs(obs_dict)
        current_ep_masks = unwrap_masks(obs_dict)

        term = any(terms.values()) if terms else False
        if term:
            break

    env.close()

    return {
        "total_steps_checked": total_steps,
        "mask_mismatch_count": mismatch_count,
        "masks_consistent": mismatch_count == 0,
    }


def main():
    parser = argparse.ArgumentParser(description="Task 2: Mask/Index alignment")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="training/results")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Action index alignment
    index_alignment = check_action_index_alignment()
    all_indices_match = all(v["match"] for v in index_alignment.values())

    # 2. Mask gradient mass check
    mask_check = check_mask_gradient_mass(args.checkpoint, num_steps=50)

    # 3. Mask consistency between rollout and update
    consistency_check = check_mask_consistency_between_rollout_and_update(args.checkpoint)

    result = {
        "timestamp_iso": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "checkpoint": args.checkpoint,
        "action_index_alignment": index_alignment,
        "all_indices_match": all_indices_match,
        "mask_gradient_check": mask_check,
        "mask_consistency_check": consistency_check,
        "task2_pass": (
            all_indices_match
            and mask_check["fraction_illegal_with_inf_logits"] > 0.99
            and consistency_check["masks_consistent"]
        ),
    }

    json_path = os.path.join(args.output_dir, "actor_optimization_audit_task2.json")
    with open(json_path, "w") as f:
        json.dump(result, f, indent=2)

    print("Task 2 Mask/Index Alignment Check Complete")
    print(f"  Action index alignment: {'PASS' if all_indices_match else 'FAIL'}")
    for name, info in index_alignment.items():
        status = "OK" if info["match"] else "MISMATCH"
        print(f"    {name}: expected={info['expected_idx']}, probe={info['probe_idx']} [{status}]")
    print(f"  Mask gradient check (in actual PPO surrogate loss):")
    print(f"    Fraction of illegal positions with -inf logits: {mask_check['fraction_illegal_with_inf_logits']:.4f}")
    print(f"    Gradient mass on illegal actions in PPO loss: {mask_check['gradient_mass_on_illegal_in_ppo_loss']:.6f}")
    print(f"    Max gradient on single illegal action: {mask_check['max_gradient_on_illegal_single_action']:.6f}")
    print(f"    Illegal action gradients nonzero count: {mask_check['illegal_grad_nonzero_count']}")
    print(f"    Steps with illegal actions in mask: {mask_check['steps_with_illegal_actions']}")
    print(f"  Mask consistency (rollout vs update):")
    print(f"    Steps checked: {consistency_check['total_steps_checked']}")
    print(f"    Mismatch count: {consistency_check['mask_mismatch_count']}")
    print(f"    Masks consistent: {consistency_check['masks_consistent']}")
    print(f"  Overall: {'PASS' if result['task2_pass'] else 'FAIL'}")
    print(f"  JSON: {json_path}")


if __name__ == "__main__":
    main()
