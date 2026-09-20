"""
Actor-Loss Frequency-Balancing Ablation — Training Runner

Runs MAPPO training with a fixed multiplier M on legal PASS/SHOT actor-loss
terms for Phase 1 (0-15k steps), then continues with M=1.0 for Phase 2
(15k-50k steps).

Seeds: 42, 123, 7, 999
Checkpoints: every 5k steps through both phases
"""
import argparse
import os
import sys
import time
from datetime import datetime, timezone

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from training.train_mappo import run_mappo_training


def main():
    parser = argparse.ArgumentParser(description="Actor-loss reweighting ablation runner")
    parser.add_argument("--seeds", type=str, default="42,123,7,999", help="Comma-separated seed list")
    parser.add_argument("--timesteps", type=int, default=50000, help="Total training timesteps per seed")
    parser.add_argument("--multiplier", type=float, default=500.0, help="Actor-loss multiplier M for Phase 1")
    parser.add_argument("--phase1-steps", type=int, default=15000, help="Phase 1 intervention window steps")
    parser.add_argument("--checkpoint-interval", type=int, default=5000, help="Checkpoint every N steps")
    parser.add_argument("--models-dir", type=str, default="training/models", help="Models directory")
    parser.add_argument("--scenario", type=str, default="academy_3_vs_1_with_keeper_onball", help="Scenario name")
    parser.add_argument("--resume", action="store_true", help="Resume from existing quarantine checkpoint")
    args = parser.parse_args()

    seeds = [int(s.strip()) for s in args.seeds.split(",")]
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    
    print("=" * 70)
    print("ACTOR-LOSS FREQUENCY-BALANCING ABLATION")
    print(f"Multiplier M: {args.multiplier}")
    print(f"Phase 1: 0-{args.phase1_steps} steps (M={args.multiplier})")
    print(f"Phase 2: {args.phase1_steps}-{args.timesteps} steps (M=1.0)")
    print(f"Checkpoints: every {args.checkpoint_interval} steps")
    print(f"Seeds: {seeds}")
    print(f"Scenario: {args.scenario}")
    print(f"Timestamp: {timestamp}")
    print("=" * 70)

    for seed in seeds:
        checkpoint_name = f"mappo_{args.scenario}_seed{seed}_actorreweight.pt"
        resume_path = None
        if args.resume:
            # Look for existing quarantine checkpoint
            candidate = os.path.join(args.models_dir, f"mappo_{args.scenario}_seed{seed}_quarantine.pt")
            if os.path.exists(candidate):
                resume_path = candidate
                print(f"\n[Seed {seed}] Resuming from {resume_path}")
            else:
                print(f"\n[Seed {seed}] No existing checkpoint found, starting fresh")

        print(f"\n[Seed {seed}] Starting training...")
        start_time = time.time()
        
        success = run_mappo_training(
            timesteps=args.timesteps,
            checkpoint_name=checkpoint_name,
            scenario=args.scenario,
            seed=seed,
            resume_path=resume_path,
            n_envs=1,
            models_dir=args.models_dir,
            enable_exploration=True,
            exploration_beta=0.03,
            mixscript_override_prob=0.0,
            mixscript_end_step=0,
            exploration_ablation_bonus=0.0,
            exploration_ablation_phase1_steps=15000,
            exploration_ablation_total_steps=30000,
            exploration_ablation_checkpoint_interval=args.checkpoint_interval,
            actor_loss_reweight_M=args.multiplier,
            actor_loss_reweight_phase1_steps=args.phase1_steps,
            actor_loss_reweight_checkpoint_interval=args.checkpoint_interval,
        )
        
        duration = time.time() - start_time
        print(f"\n[Seed {seed}] Training completed in {duration:.2f}s | Success: {success}")


if __name__ == "__main__":
    main()
