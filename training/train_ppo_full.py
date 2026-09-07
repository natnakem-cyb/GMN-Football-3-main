"""
GMN-Football-3 — Full-Match (11v11) PPO Training

Trains a PPO policy on the full-pitch ``11_vs_11`` scenario from
``src/scenarios/ScenarioRegistry.ts`` (kick-off, throw-ins, goal kicks and the
full law set are handled by the authoritative GameEngine — this script simply
points the standard PPO pipeline at the full-match scenario).

Throughput: use ``--n-envs N`` to run N parallel bridge instances (Task 1);
full-match training is impractical with a single environment.

Example (baseline proof-of-pipeline run):

    python training/train_ppo_full.py --timesteps 100000 --n-envs 4
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from training.train_ppo import run_ppo_training

FULL_MATCH_SCENARIO = "11_vs_11"


def main() -> None:
    parser = argparse.ArgumentParser(description="Train PPO on the full 11v11 match scenario")
    parser.add_argument("--timesteps", type=int, default=100_000, help="Total timesteps to train")
    parser.add_argument("--n-envs", type=int, default=4, help="Parallel environments/bridges (recommended >= 4)")
    parser.add_argument("--resume", type=str, default=None, help="Path to a PPO checkpoint to resume from")
    parser.add_argument("--checkpoint", type=str, default=None, help="Output checkpoint filename")
    parser.add_argument("--lr", type=float, default=3e-4, help="Initial learning rate")
    parser.add_argument("--lr-schedule", type=str, default="linear", choices=["constant", "linear"])
    parser.add_argument("--eval-episodes", type=int, default=10, help="Evaluation episodes at milestones")
    parser.add_argument("--opponent-difficulty", type=str, default="medium", choices=["easy", "medium", "hard", "master"])
    args = parser.parse_args()

    success = run_ppo_training(
        scenario=FULL_MATCH_SCENARIO,
        timesteps=args.timesteps,
        resume_path=args.resume,
        checkpoint_name=args.checkpoint or f"ppo_{FULL_MATCH_SCENARIO}_{args.timesteps}.zip",
        lr_schedule=args.lr_schedule,
        initial_lr=args.lr,
        eval_episodes=args.eval_episodes,
        n_envs=args.n_envs,
        opponent_difficulty=args.opponent_difficulty,
    )
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
