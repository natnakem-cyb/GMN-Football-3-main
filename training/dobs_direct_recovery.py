"""D-Obs: Direct distance recovery check.

Computes distances directly from the 127-dim observation and compares
to ground-truth distances from the engine. This tests whether the
observation contains the necessary information, independent of any
learned probe model.
"""

import json
import sys
import os
import numpy as np
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

DATASET_DIR = Path("training/results/dobs")
REPORT_DIR = Path("training/results")


def load_latest_dataset() -> list:
    files = sorted(DATASET_DIR.glob("dobs_dataset_*.json"))
    if not files:
        raise FileNotFoundError("No DOBS dataset found.")
    with open(files[-1]) as f:
        return json.load(f)


def extract_positions(obs: np.ndarray) -> dict:
    """Extract positions from 127-dim observation."""
    # Left team positions: indices 0-21 (11 players, x,y pairs)
    left_positions = []
    for i in range(11):
        x = obs[2 * i]
        y = obs[2 * i + 1]
        left_positions.append((x, y))
    
    # Right team positions: indices 44-65 (11 players, x,y pairs)
    right_positions = []
    for i in range(11):
        x = obs[44 + 2 * i]
        y = obs[44 + 2 * i + 1]
        right_positions.append((x, y))
    
    # Ball position: indices 88-90
    ball_pos = (obs[88], obs[89], obs[90])
    
    # Active player index: indices 97-107 (one-hot)
    active_idx = int(np.argmax(obs[97:108]))
    
    # Ball ownership: indices 94-96 (one-hot)
    ownership_onehot = obs[94:97]
    ownership = int(np.argmax(ownership_onehot))  # 0=none, 1=left, 2=right
    
    return {
        "left_positions": left_positions,
        "right_positions": right_positions,
        "ball_pos": ball_pos,
        "active_idx": active_idx,
        "ownership": ownership,
    }


def main():
    samples = load_latest_dataset()
    print(f"Loaded {len(samples)} samples")
    
    # Filter to samples with valid positions (not -1 padding)
    valid_samples = []
    for s in samples:
        obs = np.array(s["raw_observation"])
        parsed = extract_positions(obs)
        # Check if active player position is valid
        active_pos = parsed["left_positions"][parsed["active_idx"]]
        if active_pos[0] == -1.0 or active_pos[1] == -1.0:
            continue
        # Check if ball position is valid
        if parsed["ball_pos"][0] == -1.0:
            continue
        valid_samples.append((s, obs, parsed))
    
    print(f"Valid samples: {len(valid_samples)}")
    
    # Compute direct distances from observation
    errors_self_ball = []
    errors_teammate_ball = []
    errors_opponent_ball = []
    errors_self_goal = []
    errors_teammate_goal = []
    ownership_errors = []
    
    for s, obs, parsed in valid_samples:
        active_pos = parsed["left_positions"][parsed["active_idx"]]
        ball_pos_xy = (parsed["ball_pos"][0], parsed["ball_pos"][1])
        goal_pos = (1.0, 0.0)
        
        # Self-ball distance
        computed_d_self_ball = np.hypot(active_pos[0] - ball_pos_xy[0], active_pos[1] - ball_pos_xy[1])
        errors_self_ball.append(abs(computed_d_self_ball - s["d_self_ball"]))
        
        # Self-goal distance
        computed_d_self_goal = np.hypot(active_pos[0] - goal_pos[0], active_pos[1] - goal_pos[1])
        errors_self_goal.append(abs(computed_d_self_goal - s["d_self_goal"]))
        
        # Teammate-ball distance (nearest teammate to ball)
        teammates = [i for i in range(11) if i != parsed["active_idx"] and parsed["left_positions"][i][0] != -1.0]
        if teammates:
            min_teammate_ball = min(
                np.hypot(parsed["left_positions"][i][0] - ball_pos_xy[0], parsed["left_positions"][i][1] - ball_pos_xy[1])
                for i in teammates
            )
            errors_teammate_ball.append(abs(min_teammate_ball - s["d_teammate_ball"]))
        
        # Opponent-ball distance (nearest opponent to ball)
        opponents = [i for i in range(11) if parsed["right_positions"][i][0] != -1.0]
        if opponents:
            min_opponent_ball = min(
                np.hypot(parsed["right_positions"][i][0] - ball_pos_xy[0], parsed["right_positions"][i][1] - ball_pos_xy[1])
                for i in opponents
            )
            errors_opponent_ball.append(abs(min_opponent_ball - s["d_opponent_ball"]))
        
        # Teammate-goal distance (nearest teammate to goal)
        if teammates:
            min_teammate_goal = min(
                np.hypot(parsed["left_positions"][i][0] - goal_pos[0], parsed["left_positions"][i][1] - goal_pos[1])
                for i in teammates
            )
            errors_teammate_goal.append(abs(min_teammate_goal - s["d_teammate_goal"]))
        
        # Ownership
        expected_ownership = 0
        if s["ball_owner_id"] is None:
            expected_ownership = 0
        elif s["ball_owner_id"].startswith("left"):
            expected_ownership = 1
        elif s["ball_owner_id"].startswith("right"):
            expected_ownership = 2
        ownership_errors.append(1 if parsed["ownership"] == expected_ownership else 0)
    
    results = {
        "self_ball_mae": round(float(np.mean(errors_self_ball)), 4),
        "self_ball_max_error": round(float(np.max(errors_self_ball)), 4),
        "teammate_ball_mae": round(float(np.mean(errors_teammate_ball)), 4) if errors_teammate_ball else None,
        "opponent_ball_mae": round(float(np.mean(errors_opponent_ball)), 4) if errors_opponent_ball else None,
        "self_goal_mae": round(float(np.mean(errors_self_goal)), 4),
        "self_goal_max_error": round(float(np.max(errors_self_goal)), 4),
        "teammate_goal_mae": round(float(np.mean(errors_teammate_goal)), 4) if errors_teammate_goal else None,
        "ownership_accuracy": round(float(np.mean(ownership_errors)), 4),
        "n_samples": len(valid_samples),
    }
    
    print("\n=== Direct Distance Recovery ===")
    for k, v in results.items():
        print(f"  {k}: {v}")
    
    out_path = REPORT_DIR / "dobs_direct_recovery.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[OUTPUT] Wrote {out_path}")


if __name__ == "__main__":
    main()
