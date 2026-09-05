# IPPO vs. MAPPO: Credit-Assignment Architecture Comparison

## Executive Summary
This report compares Independent PPO (IPPO) and Multi-Agent PPO with a Centralized Critic (MAPPO) in cooperative multi-agent football scenarios. It describes the mathematical, architectural, and credit-assignment differences between the two approaches, and explains why this repository implements MAPPO with a Deep Sets centralized critic.

---

## 1. Mathematical Formulation

### IPPO Value Function (Decentralized)
In IPPO, each agent $i$ updates its policy using Generalized Advantage Estimation (GAE) computed from a decentralized local value function:
$$V_i(o_i^t) \approx \mathbb{E}_{\pi} \left[ \sum_{k=0}^{\infty} \gamma^k r^{t+k} \;\middle|\; o_i^t \right]$$
The resulting temporal difference (TD) error at timestep $t$ is:
$$\delta_i^t = r^t + \gamma V_i(o_i^{t+1}) - V_i(o_i^t)$$

### MAPPO Value Function (Centralized Joint State)
In MAPPO, the centralized critic conditions on the joint observation state $s^t = (o_1^t, o_2^t, \dots, o_N^t)$:
$$V_{\text{joint}}(s^t) \approx \mathbb{E}_{\boldsymbol{\pi}} \left[ \sum_{k=0}^{\infty} \gamma^k r_{\text{team}}^{t+k} \;\middle|\; s^t \right]$$
With TD error:
$$\delta_{\text{MAPPO}}^t = r_{\text{team}}^t + \gamma V_{\text{joint}}(s^{t+1}) - V_{\text{joint}}(s^t)$$

---

## 2. Architectural Differences

### A. Partial Observability and Non-Stationarity
In a multi-agent football environment, agent $i$'s local observation $o_i$ contains relative ego-centric coordinates and local teammate vectors. Crucial tactical signals (e.g. whether teammate $j$ has broken through the offside line, goalkeeper positioning, open passing lanes) reside in the global state $s$, not within $o_i$ alone.

For IPPO, the decentralized critic $V_i(o_i^t)$ attempts to predict team rewards without full visibility into teammates' intentions, which can increase variance in the value target and advantage estimates $\hat{A}_i^t$.

For MAPPO, the centralized critic observes the joint state, which removes this information bottleneck.

### B. Credit Attribution for Cooperative Actions
When a team reward is shared (e.g. $+1.0$ for goal), cooperative actions like passes create a credit-assignment challenge:

- Under IPPO, the advantage for the passing agent is computed from the local value function, which does not immediately observe the receiving agent's subsequent success. The credit signal can be delayed and attenuated over the intervening steps.
- Under MAPPO, the centralized critic observes the full joint state and can assign immediate advantage when the pass creates a high-quality opportunity.

### C. Permutation Invariance & Parameter Scaling
Standard centralized critics flatten all agent observations into a single dense vector $(N \times D_{\text{obs}})$, which leads to parameter explosion and poor generalization when moving between 3v1 ($3 \times 127 = 381$), 5v5 ($5 \times 127 = 635$), and 11v11 ($11 \times 127 = 1397$).

**GMN-Football-3 Solution**: We introduced the **Deep Sets Centralized Critic** with permutation-invariant dual aggregation:
$$\phi(s) = \left[ \frac{1}{N} \sum_{i=1}^N f(o_i), \; \max_{i=1}^N f(o_i) \right]$$
$$V(s) = g(\phi(s))$$
This allows $O(1)$ parameter complexity regardless of team size while preserving full permutation equivariance.

---

## 3. Comparison Summary Matrix

| Metric / Dimension | Independent PPO (IPPO) | Centralized Critic MAPPO (Deep Sets) |
| :--- | :--- | :--- |
| **Critic Input** | Local $o_i \in \mathbb{R}^{127}$ | Joint $\{o_1, \dots, o_N\} \in \mathbb{R}^{N \times 127}$ |
| **Environment Stationarity** | Non-stationary from agent's frame | Stationary under joint state-value $V(s)$ |
| **Pass / Assist Credit Attribution** | Delayed & attenuated over $\Delta t$ steps | Immediate advantage via $V(s)$ shift |
| **Lazy Agent Risk** | Higher (requires careful reward/credit design) | Lower (joint critic observes cooperation directly) |
| **Scalability across 3v1, 5v5, 11v11** | $O(1)$ network size, but high sample complexity | $O(1)$ parameter scaling via Deep Sets pooling |
| **Measured Goal Rate / Convergence** | UNMEASURED (requires local environment run) | UNMEASURED (requires local environment run) |

---

## 4. Implementation Notes

1. **Permutation-Invariant Pooling**: Integrated Deep Sets pooling in `mappo_networks.py` so the centralized critic trains effectively across 3v1, 5v5, and 11v11 without retraining flat heads.
2. **Role Differentiation**: Added 12-dimensional one-hot role embeddings to the observation space (`simple115_v3_role`), enabling actors and critics to distinguish wingers, strikers, and midfielders.
3. **Shared Trajectory Buffers**: Maintained unified rollout tensors `(num_steps, num_agents, obs_dim)` in `mappo_rollout.py` to ensure synchronous GAE computation across the team.
