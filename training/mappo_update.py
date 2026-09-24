"""
GMN-Football-3 — MAPPO (Multi-Agent PPO) Update Step
Computes clipped surrogate policy loss, centralized value loss, and entropy regularization.
Performs mini-batch gradient descent for shared actor and centralized critic.
"""

from typing import Dict, Any, Optional
import numpy as np
import torch
import torch.nn as nn

from training.mappo_networks import SharedActor, CentralizedCritic


def ppo_update(
    actor: SharedActor,
    critic: CentralizedCritic,
    actor_opt: torch.optim.Optimizer,
    critic_opt: torch.optim.Optimizer,
    buffer: Dict[str, np.ndarray],
    advantages: np.ndarray,
    returns: np.ndarray,
    clip_range: float = 0.2,
    n_epochs: int = 4,
    batch_size: int = 256,
    value_coef: float = 0.5,
    entropy_coef: float = 0.01,
    max_grad_norm: float = 0.5,
    onball_football_entropy_bonus: float = 0.0,
    actor_loss_reweight_M: float = 1.0,
) -> Dict[str, float]:
    """
    Performs PPO policy and value updates for MAPPO.

    Applies the same action-legality masks the policy sampled under during
    rollout (buffer["action_masks"], shape (T, num_agents, action_dim)) when
    re-evaluating the policy, so the stored log-probs stay a valid importance
    ratio. If the buffer has no masks (older rollouts), the update falls back
    to unmasked evaluation.

    Args:
        actor: Shared policy network
        critic: Centralized critic network
        actor_opt: Optimizer for actor
        critic_opt: Optimizer for critic
        buffer: Rollout trajectory dictionary
        advantages: GAE advantages array of shape (T,)
        returns: GAE returns array of shape (T,)
        clip_range: PPO clipping epsilon (default 0.2)
        n_epochs: Optimization epochs per rollout (default 4)
        batch_size: Mini-batch sample size (default 256)
        value_coef: Critic loss weight (default 0.5)
        entropy_coef: Policy entropy bonus weight (default 0.01)
        max_grad_norm: Maximum gradient norm clipping (default 0.5)

    Returns:
        Dictionary of mean loss and diagnostic metrics across mini-batches.
    """
    T, num_agents, obs_dim = buffer["local_obs"].shape

    # Flatten (T, num_agents, ...) -> (T*num_agents, ...), agent-major within
    # each timestep — matches collect_rollout's storage order exactly.
    flat_obs = buffer["local_obs"].reshape(T * num_agents, obs_dim)
    use_graphs = bool(getattr(actor, "requires_graph_observations", False))
    flat_actor_graphs = None
    flat_critic_graphs = None
    if use_graphs:
        graph_steps = buffer.get("graph_observations")
        if graph_steps is None or len(graph_steps) != T:
            raise ValueError("GNN PPO update requires graph_observations for every rollout step")
        flat_actor_graphs = [graph_steps[t][a] for t in range(T) for a in range(num_agents)]
        flat_critic_graphs = [graph_steps[t][0] for t in range(T) for _ in range(num_agents)]
    flat_actions = buffer["actions"].reshape(T * num_agents)
    flat_old_logprobs = buffer["logprobs"].reshape(T * num_agents)

    # Flatten legality masks in the same agent-major order (None = unmasked).
    flat_masks_t: Optional[torch.Tensor] = None
    if buffer.get("action_masks") is not None:
        mask_buffer = np.asarray(buffer["action_masks"])
        if mask_buffer.shape[:2] == (T, num_agents):
            flat_masks = mask_buffer.reshape(T * num_agents, -1)
            flat_masks_t = torch.from_numpy(flat_masks).bool()

    # Support both shared (T,) and per-agent (T, num_agents) advantage/return shapes.
    if advantages.ndim == 2:
        # Per-agent advantages (e.g. rondo asymmetric rewards): flatten directly.
        flat_advantages = advantages.reshape(-1)
        flat_returns = returns.reshape(-1)
    else:
        # Shared advantages (standard MAPPO): broadcast to every agent.
        flat_advantages = np.repeat(advantages, num_agents)
        flat_returns = np.repeat(returns, num_agents)

    # Normalize advantages — standard PPO practice
    adv_std = flat_advantages.std()
    adv_mean = flat_advantages.mean()
    flat_advantages = (flat_advantages - adv_mean) / (adv_std + 1e-8)

    # Convert arrays to tensors (zero-copy when possible)
    obs_t = torch.from_numpy(flat_obs).float()
    actions_t = torch.from_numpy(flat_actions).long()
    old_logprobs_t = torch.from_numpy(flat_old_logprobs).float()
    advantages_t = torch.from_numpy(flat_advantages).float()
    returns_t = torch.from_numpy(flat_returns).float()
    
    # 3D joint observations for scalable permutation-invariant critic (Deep Sets)
    if not use_graphs:
        joint_obs_repeated = np.repeat(buffer["local_obs"], num_agents, axis=0)
        joint_obs_t = torch.from_numpy(joint_obs_repeated).float()

    n_samples = T * num_agents
    metrics = {"policy_loss": [], "value_loss": [], "entropy": [], "approx_kl": []}
    surrogate_loss_share_M = []
    surrogate_loss_share_M1 = []

    for _ in range(n_epochs):
        indices = np.random.permutation(n_samples)
        for start in range(0, n_samples, batch_size):
            batch_idx = indices[start : start + batch_size]

            batch_masks = (
                flat_masks_t[batch_idx] if flat_masks_t is not None else None
            )
            if use_graphs:
                dist = actor([flat_actor_graphs[int(i)] for i in batch_idx], batch_masks)
            else:
                dist = actor(obs_t[batch_idx], batch_masks)
            new_logprobs = dist.log_prob(actions_t[batch_idx])
            entropy = dist.entropy().mean()

            # Targeted on-ball football entropy bonus (Form A): add extra entropy
            # from the marginal distribution over legal PASS/SHOT actions only.
            # This does not change the base reward, GAE, or mask logic.
            football_entropy_bonus = 0.0
            if onball_football_entropy_bonus > 0.0 and batch_masks is not None:
                pass_shot_legal = batch_masks[:, 9:13].any(dim=-1)  # indices 9,10,11,12
                if pass_shot_legal.any():
                    # Use the already-computed distribution; extract marginal over PASS/SHOT.
                    # Do NOT re-run actor with a restricted mask — that can trigger the
                    # fail-closed all-illegal check in states where PASS/SHOT are not legal.
                    football_probs = dist.probs.detach().clone()
                    # Zero out non-football actions and renormalize only for rows where
                    # football is legal; rows without legal football actions get 0 bonus.
                    football_mask = batch_masks[:, 9:13].float()  # (batch, 4)
                    marginal = football_probs[:, 9:13] * football_mask
                    marginal_sum = marginal.sum(dim=-1, keepdim=True).clamp_min(1e-8)
                    marginal = marginal / marginal_sum
                    # Entropy of the marginal, averaged over rows with legal football
                    legal_marginal = marginal[pass_shot_legal]
                    football_entropy = -torch.sum(
                        legal_marginal * torch.log(legal_marginal.clamp(min=1e-8)),
                        dim=-1
                    ).mean()
                    football_entropy_bonus = float(onball_football_entropy_bonus) * float(football_entropy.item())

            ratio = torch.exp(new_logprobs - old_logprobs_t[batch_idx])
            surr1_base = ratio * advantages_t[batch_idx]
            surr2_base = torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range) * advantages_t[batch_idx]
            surr1 = surr1_base.clone()
            surr2 = surr2_base.clone()

            # Actor-loss reweighting for legal PASS/SHOT transitions (M = 1.0 = no reweighting).
            # This is an actor-only multiplier; the critic loss below is unaffected.
            # FIX: gate on the action ACTUALLY SELECTED being PASS (9,10,11) or SHOT (12),
            # not just on PASS/SHOT being legal at the state. The original buggy gate fired
            # on nearly all on-ball transitions because PASS/SHOT are legal at >=99% of
            # on-ball frames, upweighting the wrong transitions.
            is_pass_shot_batch = None
            if actor_loss_reweight_M != 1.0 and batch_masks is not None:
                selected_action = actions_t[batch_idx]
                # PASS indices: 9, 10, 11; SHOT index: 12 (per ActionMapping)
                is_pass_shot_selected = (selected_action >= 9) & (selected_action <= 12)
                selected_legal = batch_masks.gather(1, selected_action.unsqueeze(-1)).squeeze(-1)
                is_pass_shot = is_pass_shot_selected & selected_legal
                is_pass_shot_batch = is_pass_shot
                if is_pass_shot.any():
                    surr1 = surr1 + (actor_loss_reweight_M - 1.0) * surr1 * is_pass_shot.float()
                    surr2 = surr2 + (actor_loss_reweight_M - 1.0) * surr2 * is_pass_shot.float()

            policy_loss = -torch.min(surr1, surr2).mean()

            # Surrogate-loss share by action class (instrumentation for actor-loss reweighting).
            # share = sum(|per-sample surrogate| for PASS/SHOT-selected-and-legal) /
            #         sum(|per-sample surrogate| over all samples)
            with torch.no_grad():
                surr_per_sample_M = torch.abs(torch.min(surr1, surr2))
                surr_per_sample_M1 = torch.abs(torch.min(surr1_base, surr2_base))
                if is_pass_shot_batch is not None and is_pass_shot_batch.any() and surr_per_sample_M.sum() > 0:
                    share_M = (surr_per_sample_M * is_pass_shot_batch.float()).sum() / surr_per_sample_M.sum()
                    share_M1 = (surr_per_sample_M1 * is_pass_shot_batch.float()).sum() / surr_per_sample_M1.sum()
                else:
                    share_M = torch.tensor(0.0, device=surr_per_sample_M.device)
                    share_M1 = torch.tensor(0.0, device=surr_per_sample_M1.device)
                surrogate_loss_share_M.append(share_M.item())
                surrogate_loss_share_M1.append(share_M1.item())

            # Pass 3D tensor (batch_size, num_agents, obs_dim) to CentralizedCritic
            if use_graphs:
                values_pred = critic([flat_critic_graphs[int(i)] for i in batch_idx])
            else:
                values_pred = critic(joint_obs_t[batch_idx])
            value_loss = ((values_pred - returns_t[batch_idx]) ** 2).mean()
            # NOTE (audit P1, Issue 6): the critic input `joint_obs_t` is the same
            # per-timestep state repeated once per agent, so the critic loss is
            # effectively computed over T*num_agents samples. For SHARED returns
            # ((T,) shape above) the repeats are identical (state, return) pairs,
            # so the loss scale is invariant to num_agents (mean of identical
            # duplicates = mean of the originals). For PER-AGENT returns ((T,
            # num_agents) shape) the critic sees the same state with different
            # targets from all agents — the loss gradient is then agent-count
            # weighted and the state-only critic cannot disambiguate the targets.
            # A "mean over agents before critic loss" normalization would fix the
            # weighting but also changes what the critic is trained to predict
            # (state value vs per-agent value) — that is a semantic redesign, not
            # a one-line fix, so it is deferred (documented, not applied).

            loss = policy_loss + value_coef * value_loss - entropy_coef * entropy - football_entropy_bonus

            actor_opt.zero_grad()
            critic_opt.zero_grad()
            loss.backward()

            if max_grad_norm is not None:
                nn.utils.clip_grad_norm_(actor.parameters(), max_grad_norm)
                nn.utils.clip_grad_norm_(critic.parameters(), max_grad_norm)

            actor_opt.step()
            critic_opt.step()

            with torch.no_grad():
                approx_kl = (old_logprobs_t[batch_idx] - new_logprobs).mean().item()

            metrics["policy_loss"].append(policy_loss.item())
            metrics["value_loss"].append(value_loss.item())
            metrics["entropy"].append(entropy.item())
            metrics["approx_kl"].append(approx_kl)

    out = {k: float(np.mean(v)) for k, v in metrics.items()}
    out["surrogate_loss_share_M"] = float(np.mean(surrogate_loss_share_M)) if surrogate_loss_share_M else 0.0
    out["surrogate_loss_share_M1"] = float(np.mean(surrogate_loss_share_M1)) if surrogate_loss_share_M1 else 0.0
    return out
