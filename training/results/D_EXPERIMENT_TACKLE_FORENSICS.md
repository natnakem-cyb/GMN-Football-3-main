# Experiment D — Tackle Exploitation Forensic Investigation

**Date:** 2026-09-17  
**HEAD at creation:** `029c459`  
**Status:** In progress — probe-only, no training  
**Objective:** Determine and eliminate the mechanism that makes tackle-related behavior an attractive policy strategy despite producing zero goals.  

---

## 1. Why D is now tackle-focused

Experiment B closed as a negative result: **0.0% final deterministic goal rate** across all four seeds.  
Experiment C closed with the same finding: exploration bonus does not improve productive behavior, and seed 123 developed pathological tackle spam (5.14 tackles/episode).

The recurrence across B and C elevates tackle spam from an isolated seed failure to a **repeated behavioral failure signature**:

| Experiment | Seed(s) with tackle spam | Tackles per episode |
|------------|--------------------------|---------------------|
| B | 7 | 9.48 |
| B | 999 | 42.64 |
| C | 123 | 5.14 |

Because this behavior is reproducible across experiments and seeds, the next step must be forensic: identify the causal mechanism before any intervention.

---

## 2. Exit criterion for D

D is complete when the investigation can state, with evidence:

1. **Which reward component(s)** contribute to tackle attractiveness (e.g., tackle reward, turnover reward, exploration bonus, penalty avoidance).
2. **Which transition semantics** make tackles appear attractive (e.g., successful vs unsuccessful tackle, possession change, ball displacement, foul/card).
3. **Whether the mechanism is** in reward semantics, engine transition semantics, possession/turnover accounting, action masking, or bridge telemetry.
4. **What single targeted intervention** would eliminate the attractive mechanism without breaking other behavior.

**D does not end until the mechanism is established.** Do not introduce an intervention on speculation.

---

## 3. Investigation scope

### 3.1 Reward semantics

- Instrument the reward adapter to emit per-component rewards for every transition, grouped by action taken.
- Decompose total episode reward into:
  - Tackle-related rewards (explicit tackle reward, tackle-triggered bonuses)
  - Possession rewards (ball recovered, possession retained)
  - Turnover rewards (forced turnovers, interceptions credited to defender)
  - Exploration bonuses (whether tackle actions received exploration bonus)
  - Penalties (fouls, cards, illegal tackles)
  - Terminal rewards (goal, concede, draw)
- Compute the **proportion of total episode reward attributable to tackle behavior** for spam vs non-spam seeds.

### 3.2 Engine transition semantics

- For each tackle action, log the engine event code and terminal reward:
  - `event_code` values: tackle successful, tackle failed, foul, card, pass intercepted, turnover, etc.
  - `reward_before_terminal` at the terminal tick after tackle
  - Whether the tackle led to possession change (ball ownership vector before/after)
  - Whether the tackle created a turnover (ball carrier change)
  - Whether the tackle displaced the ball (ball position delta)
- Identify whether tackles are being reinforced by **actual game-state improvement** (possession change, turnover) or by **reward artifacts** (sparse positive reward in a otherwise negative landscape).

### 3.3 Possession and turnover accounting

- Verify that `FootballMetricsTracker` correctly attributes:
  - Tackles that result in possession change vs tackles that do not
  - Tackles that result in turnover vs tackles that result in foul
  - Tackles attempted by the controlled agent vs tackles by opponents
- Compare ground-truth bridge telemetry (`EPISODE_STATS` frame) with tracker-computed metrics for episodes with high tackle counts. Look for discrepancies that could indicate the tracker is misattributing outcomes.

### 3.4 Action masking

- Determine whether tackle actions are **always legal** or only legal in specific states.
- If tackle is always masked as legal, the agent may be selecting it simply because it is the only action that never triggers a negative mask signal.
- Inspect `unwrap_masks` output for states preceding tackle actions. Check for:
  - All-ones mask fallback (bridge failure masking gap)
  - Missing tackle-bit masking (tackle always allowed)
  - Shot/pass mask suppression (agent cannot see productive actions)

### 3.5 Bridge telemetry

- Inspect the bridge’s action-legality mask for episodes with tackle spam.
- Determine whether the bridge is sending:
  - Correct per-state masks
  - Stale masks that default to all-ones
  - Masks that suppress shot/pass while leaving tackle enabled
- Correlate mask patterns with the first tackle in an episode. If the first tackle occurs in a state where shot/pass are masked out, the behavior is mask-driven, not reward-driven.

---

## 4. Experimental procedure

### 4.1 Re-eval with full instrumentation

Re-run the final comprehensive deterministic eval on the four Experiment B checkpoints (`mappo_ac3v1_seed{42,123,7,999}_200k_B`) with:

- 50 episodes per seed (final eval protocol)
- Full per-transition logging: action taken, reward components, event code, ball ownership, mask vector
- Output: per-episode JSON sidecar with transition-level detail

### 4.2 Episode-level aggregation

For each episode, compute:

| Metric | Definition |
|--------|------------|
| `tackle_count` | Number of times action index 16 (SLIDING) was taken |
| `tackle_reward_sum` | Sum of `reward_before_terminal` on ticks where tackle was taken |
| `tackle_event_codes` | Histogram of event codes following tackle actions |
| `tackle_possession_changes` | Number of tackles where ball ownership flipped to controlled agent |
| `tackle_turnovers_created` | Number of tackles where `turnover` event fired |
| `tackle_fouls` | Number of tackles where foul/card event fired |
| `tackle_reward_proportion` | `tackle_reward_sum / total_episode_reward` |
| `first_tackle_step` | Step of first tackle in episode |
| `first_tackle_mask` | Mask vector at first tackle state |

### 4.3 Seed-level comparison

Aggregate episode metrics across seeds and compare:

- Spam seeds (7, 999) vs non-spam seeds (42, 123)
- Identify which metrics differ systematically

### 4.4 Mechanism attribution

Based on the aggregated data, classify the dominant mechanism:

| Mechanism | Diagnostic signature |
|-----------|---------------------|
| Reward-driven | Tackle actions have positive or less-negative `reward_before_terminal` than non-tackle actions; tackle reward proportion > 0 |
| Transition-driven | Tackles produce actual possession changes or turnovers at high rate |
| Mask-driven | First tackle occurs when shot/pass masks are suppressed; tackle mask is always enabled |
| Penalty-avoidance | Tackle reward is negative but less negative than alternative actions (idle, movement) |
| Exploration-driven | Tackle actions received disproportionate exploration bonus |

Only one mechanism should be dominant. If multiple mechanisms are present, identify the **primary attractor** (the one that explains the highest proportion of tackle reward).

---

## 5. Intervention design constraint

Once the mechanism is identified, the D intervention must:

1. **Change exactly one causal variable** related to the identified mechanism.
2. **Not edit reward shaping or engine code** if the mechanism is in masking or telemetry (those are probe-only repairs).
3. **Be reversible** so that B can be re-run with the fix in place.
4. **Be documented** in `EXPERIMENT_D_QUALIFICATION.md` with before/after metrics on a single held-out seed.

---

## 6. Deliverables

| Document | Purpose |
|----------|---------|
| `training/results/D_EXPERIMENT_TACKLE_FORENSICS.md` | Investigation plan (this document) |
| `training/results/EXPERIMENT_D_QUALIFICATION.md` | Mechanism attribution, intervention design, exit criteria |
| `training/models/tackle_forensics_{seed}.json` | Per-seed per-episode transition logs |
| `training/results/tackle_forensics_summary.csv` | Aggregated episode-level metrics for analysis |

---

## 7. Provenance

- B closed as negative: `training/results/EXPERIMENT_B_PROVENANCE.md`
- B best-vs-final gap: `training/results/EXPERIMENT_B_HORIZON_200k.md`
- C report: `training/results/EXPERIMENT_C_EXPLORATION_ABLATION_100k.md`
- Experimental hierarchy: `training/experimental_hierarchy.md`
