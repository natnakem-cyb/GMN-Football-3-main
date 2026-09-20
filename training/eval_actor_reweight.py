"""
Evaluate all actor-reweight checkpoints using canonical protocol.

Canonical protocol:
- Deterministic eval (argmax)
- base_seed=500000
- 50 episodes per checkpoint
- Scenario: academy_3_vs_1_with_keeper_onball
"""
import argparse
import csv
import os
import sys
import glob

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from training.eval_progress import evaluate_checkpoint_progress


def main():
    parser = argparse.ArgumentParser(description="Evaluate all actor-reweight checkpoints")
    parser.add_argument("--models-dir", type=str, default="training/models")
    parser.add_argument("--scenario", type=str, default="academy_3_vs_1_with_keeper_onball")
    parser.add_argument("--seeds", type=str, default="42,123,7,999")
    parser.add_argument("--base-seed", type=int, default=500000)
    parser.add_argument("--num-episodes", type=int, default=50)
    parser.add_argument("--output-csv", type=str, default="training/results/actor_reweight_eval_summary.csv")
    args = parser.parse_args()

    seeds = [int(s.strip()) for s in args.seeds.split(",")]
    
    # Find all actorreweight checkpoints
    checkpoints = []
    for seed in seeds:
        # Pattern for intermediate checkpoints: _actorreweight_{step}.pt
        pattern = os.path.join(args.models_dir, f"mappo_{args.scenario}_seed{seed}_actorreweight_*.pt")
        for path in glob.glob(pattern):
            basename = os.path.basename(path)
            # Extract step number
            step_str = basename.split("_actorreweight_")[1].replace(".pt", "")
            if step_str.isdigit():
                step = int(step_str)
                checkpoints.append((seed, step, path))
        
        # Also check for final checkpoint: _actorreweight.pt (no step number)
        final_path = os.path.join(args.models_dir, f"mappo_{args.scenario}_seed{seed}_actorreweight.pt")
        if os.path.exists(final_path):
            # Try to read timesteps from checkpoint
            try:
                ckpt = torch.load(final_path, map_location="cpu")
                step = ckpt.get("timesteps", 50000)
                checkpoints.append((seed, step, final_path))
            except Exception:
                checkpoints.append((seed, 50000, final_path))
    
    # Sort by seed, then step
    checkpoints.sort(key=lambda x: (x[0], x[1]))
    
    print(f"Found {len(checkpoints)} checkpoints to evaluate")
    print(f"Seeds: {seeds}")
    print(f"Scenario: {args.scenario}")
    print(f"Episodes per eval: {args.num_episodes}")
    print()
    
    results = []
    for seed, step, path in checkpoints:
        print(f"Evaluating seed={seed}, step={step}, path={path}")
        try:
            row = evaluate_checkpoint_progress(
                checkpoint_path=path,
                scenario=args.scenario,
                algorithm="MAPPO",
                step=step,
                learning_rate=3e-4,
                num_episodes=args.num_episodes,
                deterministic=True,
                base_seed=args.base_seed,
                force_reeval=True,
            )
            results.append({
                "seed": seed,
                "step": step,
                "checkpoint": os.path.basename(path),
                "goal_rate_pct": row.get("goal_rate_pct", 0.0),
                "mean_reward": row.get("mean_reward", 0.0),
                "shots_per_ep": row.get("shots_per_ep", 0.0),
                "passes_per_ep": row.get("passes_per_ep", 0.0),
                "pass_shot_rate_pct": row.get("pass_shot_rate_pct", 0.0),
                "non_scoring_episode_rate_pct": row.get("non_scoring_episode_rate_pct", 0.0),
                "turnovers_conceded_per_ep": row.get("turnovers_conceded_per_ep", 0.0),
            })
            print(f"  -> goal_rate={row.get('goal_rate_pct', 0.0):.1f}%, "
                  f"passes_per_ep={row.get('passes_per_ep', 0.0):.2f}, "
                  f"shots_per_ep={row.get('shots_per_ep', 0.0):.2f}, "
                  f"PASS+SHOT rate={row.get('pass_shot_rate_pct', 0.0):.2f}%")
        except Exception as e:
            print(f"  -> ERROR: {e}")
        print()
    
    # Save summary CSV
    if results:
        fieldnames = ["seed", "step", "checkpoint", "goal_rate_pct", "mean_reward",
                      "shots_per_ep", "passes_per_ep", "pass_shot_rate_pct",
                      "non_scoring_episode_rate_pct", "turnovers_conceded_per_ep"]
        with open(args.output_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
        print(f"Summary saved to {args.output_csv}")
    
    # Print summary table
    print("\n" + "="*100)
    print(f"{'Seed':<8} {'Step':<8} {'Goal%':>8} {'Pass/Ep':>10} {'Shot/Ep':>10} {'PASS+SHOT%':>12} {'MeanRew':>10}")
    print("="*100)
    for r in results:
        print(f"{r['seed']:<8} {r['step']:<8} {float(r['goal_rate_pct']):>8.2f} {float(r['passes_per_ep']):>10.2f} "
              f"{float(r['shots_per_ep']):>10.2f} {float(r['pass_shot_rate_pct']):>12.2f} {float(r['mean_reward']):>10.4f}")
    print("="*100)


if __name__ == "__main__":
    main()
