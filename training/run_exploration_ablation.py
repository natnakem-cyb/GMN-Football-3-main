"""
Run the Form A exploration ablation for all 4 seeds.
Phase 1 (0-15k): targeted on-ball football entropy bonus = 0.50
Phase 2 (15k-30k): bonus = 0.0 (unscripted tail)
Intermediate checkpoints at 5k, 10k, 15k, 20k, 25k, 30k.
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from training.train_mappo import run_mappo_training

SCENARIO = "academy_3_vs_1_with_keeper"
SEEDS = [42, 123, 7, 999]
MODELS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "training", "models"))


def main():
    parser = argparse.ArgumentParser(description="Run Form A exploration ablation across all seeds")
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS, help="Seeds to run")
    parser.add_argument("--scenario", type=str, default=SCENARIO)
    parser.add_argument("--models-dir", type=str, default=MODELS_DIR)
    parser.add_argument("--skip-existing", action="store_true", help="Skip seeds whose 30k checkpoint already exists")
    args = parser.parse_args()

    os.makedirs(args.models_dir, exist_ok=True)
    results = []

    for seed in args.seeds:
        final_ckpt = os.path.join(args.models_dir, f"mappo_{args.scenario}_seed{seed}_expl_ablation_30000.pt")
        if args.skip_existing and os.path.exists(final_ckpt):
            print(f"[SKIP] seed={seed} — {final_ckpt} already exists")
            results.append((seed, "skipped", final_ckpt))
            continue

        print(f"\n{'='*70}")
        print(f"EXPLORATION ABLATION: seed={seed}")
        print(f"{'='*70}")
        start = time.time()
        ok = run_mappo_training(
            timesteps=30000,
            checkpoint_name=f"mappo_{args.scenario}_seed{seed}_expl_ablation_30000.pt",
            scenario=args.scenario,
            seed=seed,
            models_dir=args.models_dir,
            enable_exploration=True,
            exploration_beta=0.03,
            mixscript_override_prob=0.0,
            mixscript_end_step=0,
            exploration_ablation_bonus=0.5,
            exploration_ablation_phase1_steps=15000,
            exploration_ablation_total_steps=30000,
            exploration_ablation_checkpoint_interval=5000,
        )
        elapsed = time.time() - start
        results.append((seed, "ok" if ok else "FAILED", final_ckpt, elapsed))
        print(f"[DONE] seed={seed} status={'ok' if ok else 'FAILED'} time={elapsed:.1f}s")

    print(f"\n{'='*70}")
    print("EXPLORATION ABLATION RUN SUMMARY")
    print(f"{'='*70}")
    for row in results:
        print(f"  seed={row[0]} status={row[1]} ckpt={row[2]} time={row[3]:.1f}s")


if __name__ == "__main__":
    main()
