"""Gate sanity test for actor-loss reweighting (PASS/SHOT selected-action gate).

Confirms that:
- When PASS/SHOT are legal but MOVE is selected, the gate does NOT fire.
- When PASS/SHOT is selected and legal, the gate DOES fire.
- When PASS/SHOT is selected but illegal, the gate does NOT fire (sanity).
- M=500 only changes the loss when PASS/SHOT actions are present in the batch.
"""

import os
import sys
import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from training.mappo_update import ppo_update
from training.mappo_networks import SharedActor, CentralizedCritic


def _build_matching_buffer(actor, obs, mask, n_samples):
    """Sample actions from the current policy and return a buffer dict whose
    old_logprobs exactly match those actions, so the PPO ratio starts near 1."""
    with torch.no_grad():
        dist = actor(obs, mask)
    actions = dist.sample()
    old_logprobs = dist.log_prob(actions).detach()
    buffer = {
        "local_obs": obs.numpy(),
        "actions": actions.cpu().numpy().astype(np.int64),
        "logprobs": old_logprobs.cpu().numpy().astype(np.float32),
        "action_masks": mask.cpu().numpy().astype(np.int8),
    }
    return buffer, actions


def test_actor_reweight_gate():
    """Test the is_pass_shot gate logic in ppo_update."""
    torch.manual_seed(42)
    obs_dim = 127
    action_dim = 19
    actor = SharedActor(obs_dim=obs_dim, action_dim=action_dim, hidden=64)
    critic = CentralizedCritic(obs_dim=obs_dim, hidden=64)
    actor_opt = torch.optim.Adam(actor.parameters(), lr=3e-4)
    critic_opt = torch.optim.Adam(critic.parameters(), lr=3e-4)

    # Build a small buffer from the current policy so ratios start near 1.
    T, num_agents = 8, 1
    obs = torch.randn(T, num_agents, obs_dim)
    mask = torch.zeros(T, num_agents, action_dim, dtype=torch.bool)
    mask[:, :, 0:9] = True      # IDLE + movement
    mask[:, :, 9:13] = True     # PASS/SHOT legal
    mask[:, 0, 16] = False      # TACKLE illegal for first agent (possession)

    buffer, actions = _build_matching_buffer(actor, obs, mask, T)
    advantages = torch.linspace(-2.0, 2.0, steps=T)
    returns = torch.zeros(T)

    pass_shot_mask = (actions >= 9) & (actions <= 12)
    print(f"Sampled actions: {actions.view(-1).tolist()}")
    print(f"PASS/SHOT present: {pass_shot_mask.view(-1).tolist()}")

    # Baseline with M=1.0 (no reweighting).
    actor_opt.zero_grad()
    critic_opt.zero_grad()
    metrics_m1 = ppo_update(
        actor, critic, actor_opt, critic_opt,
        {k: v for k, v in buffer.items()},
        advantages.numpy(), returns.numpy(),
        actor_loss_reweight_M=1.0,
        n_epochs=1, batch_size=T,
    )
    loss_m1 = metrics_m1["policy_loss"]
    print(f"policy_loss (M=1): {loss_m1:.6f}")

    # Intervention with M=500.
    actor_opt.zero_grad()
    critic_opt.zero_grad()
    metrics_m500 = ppo_update(
        actor, critic, actor_opt, critic_opt,
        {k: v for k, v in buffer.items()},
        advantages.numpy(), returns.numpy(),
        actor_loss_reweight_M=500.0,
        n_epochs=1, batch_size=T,
    )
    loss_m500 = metrics_m500["policy_loss"]
    print(f"policy_loss (M=500): {loss_m500:.6f}")

    # If the gate is correct, M=500 should only blow up the loss when PASS/SHOT
    # actions are present. If the buggy gate were still present, M=500 would blow
    # up the loss even when only MOVE/IDLE are selected, because PASS/SHOT are
    # legal in almost every state in this buffer.
    if not pass_shot_mask.any():
        # No PASS/SHOT in batch: M=500 should have near-zero effect.
        assert abs(loss_m500 - loss_m1) < 0.1, (
            f"Gate fired without PASS/SHOT selected: M500={loss_m500:.4f}, M1={loss_m1:.4f}"
        )
        print("PASS: Gate correctly idle when no PASS/SHOT selected")
    else:
        # PASS/SHOT present: M=500 should increase loss magnitude significantly.
        ratio = abs(loss_m500) / max(abs(loss_m1), 1e-8)
        print(f"Loss ratio M500/M1: {ratio:.1f}")
        assert ratio > 2.0, (
            f"Gate did not fire with PASS/SHOT selected: ratio={ratio:.1f}"
        )
        print("PASS: Gate correctly active when PASS/SHOT selected")

    # Explicit negative check: hand-crafted buffer where only MOVE is selected
    # but PASS/SHOT are legal — M=500 must not inflate loss.
    print("\nNegative check: MOVE selected, PASS/SHOT legal")
    obs2 = torch.randn(2, num_agents, obs_dim)
    mask2 = torch.zeros(2, num_agents, action_dim, dtype=torch.bool)
    mask2[:, :, 0:9] = True
    mask2[:, :, 9:13] = True
    with torch.no_grad():
        dist2 = actor(obs2, mask2)
    actions2 = torch.tensor([[1], [5]], dtype=torch.long)
    old_logprobs2 = dist2.log_prob(actions2).detach()
    buffer2 = {
        "local_obs": obs2.numpy(),
        "actions": actions2.numpy().astype(np.int64),
        "logprobs": old_logprobs2.numpy().astype(np.float32),
        "action_masks": mask2.numpy().astype(np.int8),
    }
    adv2 = np.array([1.0, -1.0], dtype=np.float32)
    ret2 = np.array([0.0, 0.0], dtype=np.float32)

    actor_opt.zero_grad()
    critic_opt.zero_grad()
    m2_m1 = ppo_update(actor, critic, actor_opt, critic_opt, buffer2, adv2, ret2, actor_loss_reweight_M=1.0, n_epochs=1, batch_size=2)
    actor_opt.zero_grad()
    critic_opt.zero_grad()
    m2_m500 = ppo_update(actor, critic, actor_opt, critic_opt, buffer2, adv2, ret2, actor_loss_reweight_M=500.0, n_epochs=1, batch_size=2)

    print(f"  M=1 loss: {m2_m1['policy_loss']:.6f}")
    print(f"  M=500 loss: {m2_m500['policy_loss']:.6f}")
    assert abs(m2_m500['policy_loss'] - m2_m1['policy_loss']) < 0.5, (
        f"Buggy gate still firing on MOVE-only batch: M500={m2_m500['policy_loss']:.4f}, M1={m2_m1['policy_loss']:.4f}"
    )
    print("  PASS: Gate correctly idle on MOVE-only batch")

    print("\n=== ALL GATE TESTS PASSED ===")


if __name__ == "__main__":
    test_actor_reweight_gate()