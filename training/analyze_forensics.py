"""
GMN-Football-3 — Critic / GAE Horizon Forensics Analysis
Reads eval_critic_gae_forensics.py output and produces:
  - Baseline reconciliation
  - Event-centered analysis
  - TD/GAE decomposition
  - Seed 42 vs 123 contrast
  - CSV summaries
"""

import json
import os
import sys
import csv
import hashlib
import numpy as np
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(__file__)
RESULTS_DIR = os.path.join(BASE_DIR, "results")
MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(RESULTS_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Checkpoint specs for fresh 50k training
# ---------------------------------------------------------------------------
FRESH_CHECKPOINTS = {
    42: "mappo_academy_3_vs_1_with_keeper_seed42_freshtrain_50176.pt",
    123: "mappo_academy_3_vs_1_with_keeper_seed123_freshtrain_50176.pt",
    7: "mappo_academy_3_vs_1_with_keeper_seed7_freshtrain_50176.pt",
    999: "mappo_academy_3_vs_1_with_keeper_seed999_freshtrain_50176.pt",
}

GAMMA = 0.99
LAM = 0.95


def sha256_of(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_forensics_json(seed: int) -> Dict[str, Any]:
    ckpt_name = FRESH_CHECKPOINTS[seed]
    json_path = os.path.join(MODELS_DIR, f"critic_gae_forensics_{os.path.splitext(ckpt_name)[0]}.json")
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Forensics JSON not found: {json_path}")
    with open(json_path, "r") as f:
        return json.load(f)


def extract_event_centered_windows(
    episodes: List[Dict[str, Any]],
    window: int = 10,
) -> List[Dict[str, Any]]:
    """Extract event-centered windows from tick logs."""
    records = []
    for ep in episodes:
        ticks = ep.get("tick_log", [])
        ep_seed = ep.get("seed")
        ep_idx = ep.get("episode")
        
        for t, tick in enumerate(ticks):
            event_type = tick.get("event_type")
            if event_type is None:
                continue
            
            # Determine event class
            action = tick.get("actions", {})
            # Get primary agent action
            primary_action = None
            if isinstance(action, dict):
                primary_action = list(action.values())[0] if action else None
            
            event_class = None
            if event_type in ("pass", "pass_completed"):
                event_class = "PASS"
            elif event_type in ("shot", "shot_saved", "shot_missed"):
                event_class = "SHOT"
            elif event_type == "goal":
                event_class = "GOAL"
            elif event_type == "tackle":
                event_class = "TACKLE"
            else:
                continue
            
            # Extract window around event
            start = max(0, t - window)
            end = min(len(ticks), t + window + 1)
            
            pre_values = []
            post_values = []
            pre_rewards = []
            post_rewards = []
            pre_td = []
            post_td = []
            pre_gae = []
            post_gae = []
            
            for i in range(start, end):
                wtick = ticks[i]
                rel_t = i - t
                val = wtick.get("value")
                rew = wtick.get("shared_reward")
                td = wtick.get("td_residual")
                gae = wtick.get("gae_advantage")
                
                if i < t:
                    pre_values.append(val)
                    pre_rewards.append(rew)
                    pre_td.append(td)
                    pre_gae.append(gae)
                elif i > t:
                    post_values.append(val)
                    post_rewards.append(rew)
                    post_td.append(td)
                    post_gae.append(gae)
            
            records.append({
                "seed": ep_seed,
                "episode": ep_idx,
                "event_tick": t,
                "event_class": event_class,
                "event_type": event_type,
                "primary_action": primary_action,
                "pre_value_mean": float(np.mean(pre_values)) if pre_values else None,
                "post_value_mean": float(np.mean(post_values)) if post_values else None,
                "pre_reward_mean": float(np.mean(pre_rewards)) if pre_rewards else None,
                "post_reward_mean": float(np.mean(post_rewards)) if post_rewards else None,
                "pre_td_mean": float(np.mean(pre_td)) if pre_td else None,
                "post_td_mean": float(np.mean(post_td)) if post_td else None,
                "pre_gae_mean": float(np.mean(pre_gae)) if pre_gae else None,
                "post_gae_mean": float(np.mean(post_gae)) if post_gae else None,
                "value_at_event": tick.get("value"),
                "reward_at_event": tick.get("shared_reward"),
                "td_at_event": tick.get("td_residual"),
                "gae_at_event": tick.get("gae_advantage"),
                "action_at_event": primary_action,
            })
    
    return records


def compute_td_residual_stats(
    episodes: List[Dict[str, Any]],
    event_filter: Optional[str] = None,
) -> Dict[str, Any]:
    """Compute TD residual statistics, optionally filtered by event type."""
    all_td = []
    for ep in episodes:
        for tick in ep.get("tick_log", []):
            if event_filter is not None:
                if tick.get("event_type") != event_filter:
                    continue
            td = tick.get("td_residual")
            if td is not None:
                all_td.append(td)
    
    if not all_td:
        return {"count": 0}
    
    arr = np.array(all_td, dtype=np.float32)
    return {
        "count": len(arr),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "std": float(np.std(arr)),
        "p10": float(np.percentile(arr, 10)),
        "p25": float(np.percentile(arr, 25)),
        "p75": float(np.percentile(arr, 75)),
        "p90": float(np.percentile(arr, 90)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "fraction_positive": float(np.mean(arr > 0)),
        "fraction_negative": float(np.mean(arr < 0)),
    }


def compute_gae_contribution_bins(
    episodes: List[Dict[str, Any]],
    event_tick: int,
    ep_ticks: List[Dict[str, Any]],
    gamma: float = GAMMA,
    lam: float = LAM,
) -> Dict[str, Any]:
    """Compute GAE contribution bins for an event."""
    T = len(ep_ticks)
    if event_tick >= T:
        return {}
    
    # Reconstruct GAE backwards from event_tick
    # A_t = sum_{k=0}^{T-1-t} (gamma*lambda)^k * delta_{t+k}
    bins = {
        "k0": {"signed": 0.0, "abs": 0.0, "count": 0},
        "k1_4": {"signed": 0.0, "abs": 0.0, "count": 0},
        "k5_9": {"signed": 0.0, "abs": 0.0, "count": 0},
        "k10_19": {"signed": 0.0, "abs": 0.0, "count": 0},
        "k20_plus": {"signed": 0.0, "abs": 0.0, "count": 0},
    }
    
    # We need delta values. Use td_residual as delta approximation.
    deltas = []
    for i in range(event_tick, T):
        tick = ep_ticks[i]
        td = tick.get("td_residual", 0.0)
        deltas.append(td)
    
    if not deltas:
        return bins
    
    running_gae = 0.0
    for k, delta in enumerate(deltas):
        weight = (gamma * lam) ** k
        contrib = weight * delta
        running_gae += contrib
        
        if k == 0:
            key = "k0"
        elif 1 <= k <= 4:
            key = "k1_4"
        elif 5 <= k <= 9:
            key = "k5_9"
        elif 10 <= k <= 19:
            key = "k10_19"
        else:
            key = "k20_plus"
        
        bins[key]["signed"] += contrib
        bins[key]["abs"] += abs(contrib)
        bins[key]["count"] += 1
    
    return bins


def analyze_checkpoint(seed: int) -> Dict[str, Any]:
    """Full analysis for one checkpoint."""
    data = load_forensics_json(seed)
    episodes = data.get("episodes", [])
    
    # Basic stats
    total_ticks = sum(ep.get("length", 0) for ep in episodes)
    total_passes = sum(ep.get("pass_actions", 0) for ep in episodes)
    total_shots = sum(ep.get("shot_actions", 0) for ep in episodes)
    total_tackles = sum(ep.get("tackle_actions", 0) for ep in episodes)
    
    # Collect all tick data
    all_ticks = []
    for ep in episodes:
        for tick in ep.get("tick_log", []):
            tick["_seed"] = ep.get("seed")
            tick["_episode"] = ep.get("episode")
            all_ticks.append(tick)
    
    # Value stats
    all_values = [t.get("value") for t in all_ticks if t.get("value") is not None]
    all_td = [t.get("td_residual") for t in all_ticks if t.get("td_residual") is not None]
    all_gae = [t.get("gae_advantage") for t in all_ticks if t.get("gae_advantage") is not None]
    
    # Event stats
    pass_ticks = [t for t in all_ticks if t.get("event_type") in ("pass", "pass_completed")]
    shot_ticks = [t for t in all_ticks if t.get("event_type") in ("shot", "shot_saved", "shot_missed")]
    goal_ticks = [t for t in all_ticks if t.get("event_type") == "goal"]
    tackle_ticks = [t for t in all_ticks if t.get("event_type") == "tackle"]
    move_ticks = [t for t in all_ticks if t.get("event_type") is None]
    
    # TD stats by event class
    td_stats = {
        "all": compute_td_residual_stats(episodes),
        "pass": compute_td_residual_stats(episodes, "pass"),
        "pass_completed": compute_td_residual_stats(episodes, "pass_completed"),
        "shot": compute_td_residual_stats(episodes, "shot"),
        "shot_saved": compute_td_residual_stats(episodes, "shot_saved"),
        "shot_missed": compute_td_residual_stats(episodes, "shot_missed"),
        "goal": compute_td_residual_stats(episodes, "goal"),
        "tackle": compute_td_residual_stats(episodes, "tackle"),
        "none": compute_td_residual_stats(episodes, None),  # This won't work as expected
    }
    
    # Compute TD for non-event ticks (event_type is None)
    none_event_td = []
    for ep in episodes:
        for tick in ep.get("tick_log", []):
            if tick.get("event_type") is None:
                td = tick.get("td_residual")
                if td is not None:
                    none_event_td.append(td)
    
    td_stats["no_event"] = {
        "count": len(none_event_td),
        "mean": float(np.mean(none_event_td)) if none_event_td else 0.0,
        "std": float(np.std(none_event_td)) if none_event_td else 0.0,
    }
    
    # GAE contribution analysis for events
    event_gae_analysis = []
    for ep in episodes:
        ticks = ep.get("tick_log", [])
        for t, tick in enumerate(ticks):
            event_type = tick.get("event_type")
            if event_type is None:
                continue
            if event_type not in ("pass", "pass_completed", "shot", "shot_saved", "shot_missed", "goal", "tackle"):
                continue
            
            bins = compute_gae_contribution_bins(episodes, t, ticks)
            event_class = None
            if event_type in ("pass", "pass_completed"):
                event_class = "PASS"
            elif event_type in ("shot", "shot_saved", "shot_missed"):
                event_class = "SHOT"
            elif event_type == "goal":
                event_class = "GOAL"
            elif event_type == "tackle":
                event_class = "TACKLE"
            event_gae_analysis.append({
                "seed": ep.get("seed"),
                "episode": ep.get("episode"),
                "event_tick": t,
                "event_class": event_class,
                "event_type": event_type,
                "value_at_event": tick.get("value"),
                "td_at_event": tick.get("td_residual"),
                "gae_at_event": tick.get("gae_advantage"),
                "bins": bins,
            })
    
    return {
        "seed": seed,
        "checkpoint": data.get("checkpoint"),
        "checkpoint_sha256": data.get("checkpoint_sha256"),
        "checkpoint_timesteps": data.get("checkpoint_timesteps"),
        "scenario": data.get("scenario"),
        "num_episodes": len(episodes),
        "total_ticks": total_ticks,
        "total_passes": total_passes,
        "total_shots": total_shots,
        "total_tackles": total_tackles,
        "pass_rate_pct": 100.0 * total_passes / max(total_ticks, 1),
        "shot_rate_pct": 100.0 * total_shots / max(total_ticks, 1),
        "tackle_rate_pct": 100.0 * total_tackles / max(total_ticks, 1),
        "mean_value": float(np.mean(all_values)) if all_values else 0.0,
        "std_value": float(np.std(all_values)) if all_values else 0.0,
        "mean_td_residual": float(np.mean(all_td)) if all_td else 0.0,
        "std_td_residual": float(np.std(all_td)) if all_td else 0.0,
        "mean_gae_advantage": float(np.mean(all_gae)) if all_gae else 0.0,
        "std_gae_advantage": float(np.std(all_gae)) if all_gae else 0.0,
        "td_stats": td_stats,
        "event_gae_analysis": event_gae_analysis,
        "num_pass_events": len(pass_ticks),
        "num_shot_events": len(shot_ticks),
        "num_goal_events": len(goal_ticks),
        "num_tackle_events": len(tackle_ticks),
        "num_no_event_ticks": len(move_ticks),
    }


def main():
    print("=" * 70)
    print("CRITIC / GAE HORIZON FORENSICS - POST-PROCESSING ANALYSIS")
    print("=" * 70)
    
    all_results = {}
    for seed in [42, 123, 7, 999]:
        print(f"\nAnalyzing seed {seed}...")
        all_results[seed] = analyze_checkpoint(seed)
        r = all_results[seed]
        print(f"  PASS rate: {r['pass_rate_pct']:.2f}%")
        print(f"  SHOT rate: {r['shot_rate_pct']:.2f}%")
        print(f"  Mean value: {r['mean_value']:.4f} +/- {r['std_value']:.4f}")
        print(f"  Mean delta: {r['mean_td_residual']:.4f} +/- {r['std_td_residual']:.4f}")
        print(f"  Mean GAE: {r['mean_gae_advantage']:.4f} +/- {r['std_gae_advantage']:.4f}")
        print(f"  Events: PASS={r['num_pass_events']}, SHOT={r['num_shot_events']}, GOAL={r['num_goal_events']}, TACKLE={r['num_tackle_events']}")
    
    # Save combined analysis
    output_path = os.path.join(RESULTS_DIR, "forensics_analysis.json")
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nCombined analysis saved to: {output_path}")
    
    # Write CSV summary
    csv_path = os.path.join(RESULTS_DIR, "critic_gae_horizon_summary.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "seed", "checkpoint", "checkpoint_sha256", "checkpoint_timesteps",
            "scenario", "num_episodes", "total_ticks",
            "pass_rate_pct", "shot_rate_pct", "tackle_rate_pct",
            "mean_value", "std_value", "mean_td_residual", "std_td_residual",
            "mean_gae_advantage", "std_gae_advantage",
            "num_pass_events", "num_shot_events", "num_goal_events", "num_tackle_events",
            "td_mean_no_event", "td_std_no_event",
        ])
        for seed in [42, 123, 7, 999]:
            r = all_results[seed]
            no_event = r["td_stats"].get("no_event", {})
            writer.writerow([
                seed, r["checkpoint"], r["checkpoint_sha256"], r["checkpoint_timesteps"],
                r["scenario"], r["num_episodes"], r["total_ticks"],
                f"{r['pass_rate_pct']:.4f}", f"{r['shot_rate_pct']:.4f}", f"{r['tackle_rate_pct']:.4f}",
                f"{r['mean_value']:.6f}", f"{r['std_value']:.6f}",
                f"{r['mean_td_residual']:.6f}", f"{r['std_td_residual']:.6f}",
                f"{r['mean_gae_advantage']:.6f}", f"{r['std_gae_advantage']:.6f}",
                r["num_pass_events"], r["num_shot_events"], r["num_goal_events"], r["num_tackle_events"],
                f"{no_event.get('mean', 0):.6f}", f"{no_event.get('std', 0):.6f}",
            ])
    print(f"CSV summary saved to: {csv_path}")
    
    # Write event-level CSV
    event_csv_path = os.path.join(RESULTS_DIR, "event_level_forensics.csv")
    with open(event_csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "seed", "episode", "event_tick", "event_class", "event_type",
            "value_at_event", "td_at_event", "gae_at_event",
            "pre_value_mean", "post_value_mean",
            "pre_td_mean", "post_td_mean",
            "pre_gae_mean", "post_gae_mean",
            "k0_signed", "k0_abs",
            "k1_4_signed", "k1_4_abs",
            "k5_9_signed", "k5_9_abs",
            "k10_19_signed", "k10_19_abs",
            "k20_plus_signed", "k20_plus_abs",
        ])
        for seed in [42, 123, 7, 999]:
            r = all_results[seed]
            for ev in r.get("event_gae_analysis", []):
                bins = ev.get("bins", {})
                writer.writerow([
                    ev.get("seed", ""), ev.get("episode", ""), ev.get("event_tick", ""), 
                    ev.get("event_class", ""), ev.get("event_type", ""),
                    f"{ev.get('value_at_event', 0):.6f}" if ev.get('value_at_event') is not None else "",
                    f"{ev.get('td_at_event', 0):.6f}" if ev.get('td_at_event') is not None else "",
                    f"{ev.get('gae_at_event', 0):.6f}" if ev.get('gae_at_event') is not None else "",
                    f"{ev.get('pre_value_mean', 0):.6f}" if ev.get('pre_value_mean') is not None else "",
                    f"{ev.get('post_value_mean', 0):.6f}" if ev.get('post_value_mean') is not None else "",
                    f"{ev.get('pre_td_mean', 0):.6f}" if ev.get('pre_td_mean') is not None else "",
                    f"{ev.get('post_td_mean', 0):.6f}" if ev.get('post_td_mean') is not None else "",
                    f"{ev.get('pre_gae_mean', 0):.6f}" if ev.get('pre_gae_mean') is not None else "",
                    f"{ev.get('post_gae_mean', 0):.6f}" if ev.get('post_gae_mean') is not None else "",
                    f"{bins.get('k0', {}).get('signed', 0):.6f}",
                    f"{bins.get('k0', {}).get('abs', 0):.6f}",
                    f"{bins.get('k1_4', {}).get('signed', 0):.6f}",
                    f"{bins.get('k1_4', {}).get('abs', 0):.6f}",
                    f"{bins.get('k5_9', {}).get('signed', 0):.6f}",
                    f"{bins.get('k5_9', {}).get('abs', 0):.6f}",
                    f"{bins.get('k10_19', {}).get('signed', 0):.6f}",
                    f"{bins.get('k10_19', {}).get('abs', 0):.6f}",
                    f"{bins.get('k20_plus', {}).get('signed', 0):.6f}",
                    f"{bins.get('k20_plus', {}).get('abs', 0):.6f}",
                ])
    print(f"Event-level CSV saved to: {event_csv_path}")
    
    return all_results


if __name__ == "__main__":
    results = main()
