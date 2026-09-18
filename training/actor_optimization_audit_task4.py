"""
Actor-Optimization / Advantage-Consumption Audit — Task 4: Stale Value Check

Compares the critic checkpoint used in the probe (50k) against intermediate
checkpoints to quantify the staleness gap.

Outputs:
  - training/results/actor_optimization_audit_task4.json
"""
import argparse
import json
import os
import re
import sys
from datetime import datetime, timezone

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from training.mappo_networks import CentralizedCritic


def load_critic_state(path: str):
    ckpt = torch.load(path, map_location="cpu")
    if "critic" not in ckpt:
        return None
    return ckpt["critic"]


def compute_state_distance(state1, state2):
    """Compute L2 distance between two state dicts."""
    total_dist = 0.0
    total_params = 0
    for key in state1:
        if key in state2:
            diff = state1[key] - state2[key]
            total_dist += torch.sum(diff ** 2).item()
            total_params += diff.numel()
    return np.sqrt(total_dist), total_params


def find_intermediate_checkpoints(base_name: str, max_step: int = 50000, seed: int = 0):
    """Find intermediate checkpoints for a given base name.
    
    Excludes the seed number and the final checkpoint.
    """
    import glob, re
    pattern = f"training/models/{base_name}_*.pt"
    files = glob.glob(pattern)
    checkpoints = []
    for f in files:
        # Extract step number from filename (last number before .pt)
        basename = os.path.basename(f).replace(".pt", "")
        numbers = re.findall(r'\d+', basename)
        if numbers:
            # Filter out seed numbers (typically small, < 1000)
            # and use the largest number as the step count
            candidates = [int(n) for n in numbers if int(n) >= 1000]
            if candidates:
                step = max(candidates)
                if step < max_step:
                    checkpoints.append((step, f))
    checkpoints.sort(key=lambda x: x[0])
    return checkpoints


def main():
    parser = argparse.ArgumentParser(description="Task 4: Stale value check")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="training/results")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Extract base name from checkpoint path
    basename = os.path.basename(args.checkpoint).replace(".pt", "")
    # Remove _freshtrain_50176 suffix to get base name
    base_name = basename.replace("_freshtrain_50176", "")
    # Extract seed number
    seed_match = re.search(r'seed(\d+)', basename)
    seed = int(seed_match.group(1)) if seed_match else 0

    # Load 50k critic
    critic_50k = load_critic_state(args.checkpoint)
    if critic_50k is None:
        print("No critic found in checkpoint")
        return

    # Find intermediate checkpoints
    intermediate = find_intermediate_checkpoints(base_name, max_step=50000, seed=seed)

    distances = []
    for step, path in intermediate:
        critic_state = load_critic_state(path)
        if critic_state is None:
            continue
        l2_dist, n_params = compute_state_distance(critic_50k, critic_state)
        distances.append({
            "step": step,
            "path": path,
            "l2_distance": l2_dist,
            "n_params": n_params,
            "relative_distance": l2_dist / np.sqrt(n_params) if n_params > 0 else 0.0,
        })

    result = {
        "timestamp_iso": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "checkpoint_50k": args.checkpoint,
        "base_name": base_name,
        "intermediate_checkpoints": len(distances),
        "distances": distances,
        "max_l2_distance": max((d["l2_distance"] for d in distances), default=0.0),
        "mean_l2_distance": float(np.mean([d["l2_distance"] for d in distances])) if distances else 0.0,
    }

    json_path = os.path.join(args.output_dir, "actor_optimization_audit_task4.json")
    with open(json_path, "w") as f:
        json.dump(result, f, indent=2)

    print("Task 4 Stale Value Check Complete")
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  Intermediate checkpoints found: {len(distances)}")
    for d in distances:
        print(f"    Step {d['step']}: L2={d['l2_distance']:.6f}, relative={d['relative_distance']:.6f}")
    print(f"  Max L2 distance: {result['max_l2_distance']:.6f}")
    print(f"  Mean L2 distance: {result['mean_l2_distance']:.6f}")
    print(f"  JSON: {json_path}")


if __name__ == "__main__":
    main()
