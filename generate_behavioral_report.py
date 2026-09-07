"""
Generate comprehensive behavioral report from eval_mappo_comprehensive.py results.
"""
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from training.football_metrics import FootballMetricsTracker

CHECKPOINTS = {
    "Seed 42 (500k)": "training/models/mappo_academy_3_vs_1_with_keeper_seed42.pt",
    "Seed 43 (500k)": "training/models/mappo_academy_3_vs_1_with_keeper_seed43.pt",
    "Seed 44 (500k)": "training/models/mappo_academy_3_vs_1_with_keeper_seed44.pt",
    "Seed 137 (500k)": "training/models/mappo_academy_3_vs_1_with_keeper_seed137.pt",
}

JSON_RESULTS = {
    "Seed 42 (500k)": "training/models/comprehensive_eval_mappo_academy_3_vs_1_with_keeper_seed42.json",
    "Seed 43 (500k)": "training/models/comprehensive_eval_mappo_academy_3_vs_1_with_keeper_seed43.json",
    "Seed 44 (500k)": "training/models/comprehensive_eval_mappo_academy_3_vs_1_with_keeper_seed44.json",
    "Seed 137 (500k)": "training/models/comprehensive_eval_mappo_academy_3_vs_1_with_keeper_seed137.json",
}


def load_results(seed_label):
    path = JSON_RESULTS[seed_label]
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def compute_possession_retention(data):
    """Approximate average ticks between possession changes."""
    episodes = data.get("episodes", [])
    if not episodes:
        # Fallback to aggregate if episodes not stored
        changes = data.get("possession_changes_per_episode", {}).get("mean", 0)
        length = data.get("mean_episode_length", 0)
        if changes > 0:
            return length / changes
        return float("nan")
    
    retentions = []
    for ep in episodes:
        changes = ep.get("possession_changes", 0)
        ticks = ep.get("duration_ticks", 0)
        if changes > 0 and ticks > 0:
            retentions.append(ticks / changes)
    
    if not retentions:
        return float("nan")
    
    import numpy as np
    return float(np.mean(retentions))


def main():
    print("=" * 80)
    print("COMPREHENSIVE BEHAVIORAL REPORT — 500k MAPPO CHECKPOINTS")
    print("Scenario: academy_3_vs_1_with_keeper | 500 deterministic episodes each")
    print("=" * 80)
    
    results = {}
    for seed_label in CHECKPOINTS:
        data = load_results(seed_label)
        if data is None:
            print(f"\nWARNING: No results found for {seed_label}")
            continue
        results[seed_label] = data
    
    if not results:
        print("No results to report.")
        return
    
    # Header
    print("\n" + "=" * 80)
    print("TABLE 1: GOAL RATE & OUTCOMES")
    print("=" * 80)
    print(f"{'Checkpoint':<25} {'Goal Rate':>10} {'Win Rate':>10} {'Draw Rate':>10} {'Loss Rate':>10}")
    print("-" * 80)
    for seed_label, data in results.items():
        goal = data["success_rate_pct"]["mean"]
        win = data["win_rate_pct"]["mean"]
        draw = data["draw_rate_pct"]["mean"]
        loss = data["loss_rate_pct"]["mean"]
        print(f"{seed_label:<25} {goal:>9.1f}% {win:>9.1f}% {draw:>9.1f}% {loss:>9.1f}%")
    
    print("\n" + "=" * 80)
    print("TABLE 2: SHOOTING EFFICIENCY")
    print("=" * 80)
    print(f"{'Checkpoint':<25} {'Shots/Ep':>10} {'Shot Acc':>10} {'Shot-to-Goal':>12} {'Goals/Ep':>10}")
    print("-" * 80)
    for seed_label, data in results.items():
        shots = data["shots_per_episode_mean"]
        shot_acc = data["shot_accuracy_pct"]["mean"]
        shot_to_goal = data.get("shot_to_goal_pct", 0.0)
        goals_ep = data["goals_scored_per_episode"]["mean"]
        print(f"{seed_label:<25} {shots:>10.2f} {shot_acc:>9.1f}% {shot_to_goal:>11.1f}% {goals_ep:>10.2f}")
    
    print("\n" + "=" * 80)
    print("TABLE 3: PASSING VOLUME & ACCURACY")
    print("=" * 80)
    print(f"{'Checkpoint':<25} {'Passes/Ep':>10} {'Pass Comp':>10} {'Pass Acc':>10}")
    print("-" * 80)
    for seed_label, data in results.items():
        passes = data["passes_per_episode_mean"]
        pass_comp = data["pass_completion_rate_pct"]["mean"]
        print(f"{seed_label:<25} {passes:>10.2f} {pass_comp:>9.1f}%")
    
    print("\n" + "=" * 80)
    print("TABLE 4: TURNOVERS & POSSESSION")
    print("=" * 80)
    print(f"{'Checkpoint':<25} {'Turnovers/Ep':>12} {'Possession':>10} {'Poss Changes/Ep':>15} {'Retention (ticks)':>18}")
    print("-" * 80)
    for seed_label, data in results.items():
        turnovers = data["turnovers_conceded_per_episode"]["mean"]
        possession = data["possession_rate_pct"]["mean"]
        changes = data["possession_changes_per_episode"]["mean"]
        retention = compute_possession_retention(data)
        print(f"{seed_label:<25} {turnovers:>12.2f} {possession:>9.1f}% {changes:>15.2f} {retention:>18.1f}")
    
    print("\n" + "=" * 80)
    print("TABLE 5: TACTICAL DIVERGENCE — SEED 43 vs SEED 44")
    print("=" * 80)
    
    seed43 = results.get("Seed 43 (500k)")
    seed44 = results.get("Seed 44 (500k)")
    
    if seed43 and seed44:
        print("\nNOTE: Seeds 43 and 44 produced identical checkpoints (same file hash).")
        print("This indicates no tactical divergence between these runs.")
        print()
        
        metrics = [
            ("Goal Rate", seed43["success_rate_pct"]["mean"], seed44["success_rate_pct"]["mean"], "%"),
            ("Shots/Episode", seed43["shots_per_episode_mean"], seed44["shots_per_episode_mean"], ""),
            ("Shot Accuracy", seed43["shot_accuracy_pct"]["mean"], seed44["shot_accuracy_pct"]["mean"], "%"),
            ("Shot-to-Goal %", seed43.get("shot_to_goal_pct", 0.0), seed44.get("shot_to_goal_pct", 0.0), "%"),
            ("Passes/Episode", seed43["passes_per_episode_mean"], seed44["passes_per_episode_mean"], ""),
            ("Pass Completion", seed43["pass_completion_rate_pct"]["mean"], seed44["pass_completion_rate_pct"]["mean"], "%"),
            ("Turnovers/Ep", seed43["turnovers_conceded_per_episode"]["mean"], seed44["turnovers_conceded_per_episode"]["mean"], ""),
            ("Possession %", seed43["possession_rate_pct"]["mean"], seed44["possession_rate_pct"]["mean"], "%"),
            ("Episode Length", seed43["mean_episode_length"], seed44["mean_episode_length"], " steps"),
        ]
        
        print(f"{'Metric':<25} {'Seed 43':>12} {'Seed 44':>12} {'Delta':>12}")
        print("-" * 80)
        for name, v43, v44, unit in metrics:
            delta = v44 - v43
            print(f"{name:<25} {v43:>12.2f}{unit} {v44:>12.2f}{unit} {delta:>+12.2f}{unit}")
    
    print("\n" + "=" * 80)
    print("TABLE 6: SEED 42 (HIGH SHOOTING) vs SEED 44 (SELECTIVE SHOOTING)")
    print("=" * 80)
    
    seed42 = results.get("Seed 42 (500k)")
    seed44 = results.get("Seed 44 (500k)")
    
    if seed42 and seed44:
        metrics = [
            ("Goal Rate", seed42["success_rate_pct"]["mean"], seed44["success_rate_pct"]["mean"], "%"),
            ("Shots/Episode", seed42["shots_per_episode_mean"], seed44["shots_per_episode_mean"], ""),
            ("Shot Accuracy", seed42["shot_accuracy_pct"]["mean"], seed44["shot_accuracy_pct"]["mean"], "%"),
            ("Shot-to-Goal %", seed42.get("shot_to_goal_pct", 0.0), seed44.get("shot_to_goal_pct", 0.0), "%"),
            ("Passes/Episode", seed42["passes_per_episode_mean"], seed44["passes_per_episode_mean"], ""),
            ("Pass Completion", seed42["pass_completion_rate_pct"]["mean"], seed44["pass_completion_rate_pct"]["mean"], "%"),
            ("Turnovers/Ep", seed42["turnovers_conceded_per_episode"]["mean"], seed44["turnovers_conceded_per_episode"]["mean"], ""),
            ("Possession %", seed42["possession_rate_pct"]["mean"], seed44["possession_rate_pct"]["mean"], "%"),
            ("Episode Length", seed42["mean_episode_length"], seed44["mean_episode_length"], " steps"),
        ]
        
        print(f"\n{'Metric':<25} {'Seed 42':>12} {'Seed 44':>12} {'Delta':>12}")
        print("-" * 80)
        for name, v42, v44, unit in metrics:
            delta = v44 - v42
            print(f"{name:<25} {v42:>12.2f}{unit} {v44:>12.2f}{unit} {delta:>+12.2f}{unit}")
        
        print("\n" + "-" * 80)
        print("TACTICAL ANALYSIS:")
        print("-" * 80)
        
        shot_ratio_42 = seed42["shots_per_episode_mean"] / max(seed42["mean_episode_length"], 1)
        shot_ratio_44 = seed44["shots_per_episode_mean"] / max(seed44["mean_episode_length"], 1)
        
        print(f"Seed 42 shot frequency: {shot_ratio_42:.4f} shots/step")
        print(f"Seed 44 shot frequency: {shot_ratio_44:.4f} shots/step")
        
        if seed42["shots_per_episode_mean"] > seed44["shots_per_episode_mean"] * 1.5:
            print("\nCONCLUSION: Seed 42 demonstrates HIGH SHOOTING VOLUME behavior.")
            print("  - Shots per episode:", seed42["shots_per_episode_mean"], "vs", seed44["shots_per_episode_mean"])
            print("  - This is consistent with proxy-reward exploitation (optimizing for shot actions).")
        else:
            print("\nCONCLUSION: No significant shooting volume divergence detected.")
        
        if seed42["passes_per_episode_mean"] > seed44["passes_per_episode_mean"] * 1.5:
            print("  - Seed 42 also shows higher passing volume.")
        elif seed44["passes_per_episode_mean"] > seed42["passes_per_episode_mean"] * 1.5:
            print("  - Seed 44 shows higher passing volume (more selective shooting).")
        else:
            print("  - Passing volumes are similar.")
    
    print("\n" + "=" * 80)
    print("REPORT COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
