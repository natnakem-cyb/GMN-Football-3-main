"""D-Obs: Counterfactual and state-aliasing tests.

Constructs paired states with controlled semantic changes and tests
whether the observation changes in a way sufficient to distinguish them.
Also searches for state aliasing where observations are similar but
action-relevant variables differ.
"""

import json
import sys
import os
import numpy as np
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import pairwise_distances

DATASET_DIR = Path("training/results/dobs")
REPORT_DIR = Path("training/results")


def load_latest_dataset() -> list:
    files = sorted(DATASET_DIR.glob("dobs_dataset_*.json"))
    if not files:
        raise FileNotFoundError("No DOBS dataset found.")
    with open(files[-1]) as f:
        return json.load(f)


def extract_positions(obs: np.ndarray) -> dict:
    left_positions = []
    for i in range(11):
        x = obs[2 * i]
        y = obs[2 * i + 1]
        left_positions.append((x, y))
    
    right_positions = []
    for i in range(11):
        x = obs[44 + 2 * i]
        y = obs[44 + 2 * i + 1]
        right_positions.append((x, y))
    
    ball_pos = (obs[88], obs[89], obs[90])
    active_idx = int(np.argmax(obs[97:108]))
    ownership_onehot = obs[94:97]
    ownership = int(np.argmax(ownership_onehot))
    role_onehot = obs[115:127]
    role_idx = int(np.argmax(role_onehot))
    
    return {
        "left_positions": left_positions,
        "right_positions": right_positions,
        "ball_pos": ball_pos,
        "active_idx": active_idx,
        "ownership": ownership,
        "role_idx": role_idx,
    }


def find_counterfactual_pairs(samples: list, n_pairs: int = 50) -> list:
    """Find paired states where one semantic variable changes."""
    pairs = []
    
    # Group by scenario and approximate self-ball distance
    by_scenario = {}
    for s in samples:
        key = s["scenario_id"]
        if key not in by_scenario:
            by_scenario[key] = []
        by_scenario[key].append(s)
    
    # Ownership pairs: same geometry, different owner
    for scenario, scenario_samples in by_scenario.items():
        left_possess = [s for s in scenario_samples if s["ball_owner_id"] and s["ball_owner_id"].startswith("left")]
        no_owner = [s for s in scenario_samples if s["ball_owner_id"] is None]
        
        for _ in range(min(n_pairs // 4, len(left_possess), len(no_owner))):
            s1 = np.random.choice(left_possess)
            s2 = np.random.choice(no_owner)
            pairs.append({
                "type": "ownership",
                "s1": s1,
                "s2": s2,
                "expected_diff": "owner(s1) != owner(s2)",
            })
    
    # Teammate distance pairs: same self/ball, different teammate position
    for scenario, scenario_samples in by_scenario.items():
        if len(scenario_samples) < 2:
            continue
        for _ in range(n_pairs // 4):
            s1 = np.random.choice(scenario_samples)
            s2 = np.random.choice(scenario_samples)
            if s1["controlled_player_id"] == s2["controlled_player_id"]:
                pairs.append({
                    "type": "teammate_position",
                    "s1": s1,
                    "s2": s2,
                    "expected_diff": "teammate geometry differs",
                })
    
    # Opponent distance pairs
    for scenario, scenario_samples in by_scenario.items():
        if len(scenario_samples) < 2:
            continue
        for _ in range(n_pairs // 4):
            s1 = np.random.choice(scenario_samples)
            s2 = np.random.choice(scenario_samples)
            pairs.append({
                "type": "opponent_position",
                "s1": s1,
                "s2": s2,
                "expected_diff": "opponent geometry differs",
            })
    
    return pairs


def compute_observation_distance(s1: dict, s2: dict) -> float:
    """Compute Euclidean distance between two observations."""
    obs1 = np.array(s1["raw_observation"])
    obs2 = np.array(s2["raw_observation"])
    return float(np.linalg.norm(obs1 - obs2))


def find_state_aliases(samples: list, threshold: float = 0.1, max_pairs: int = 50) -> list:
    """Find pairs where observations are similar but action-relevant variables differ."""
    aliases = []
    obs_matrix = np.array([s["raw_observation"] for s in samples])
    
    # Compute pairwise distances (subsample for efficiency)
    n = min(len(samples), 200)
    indices = np.random.choice(len(samples), n, replace=False)
    obs_subset = obs_matrix[indices]
    
    distances = pairwise_distances(obs_subset, metric="euclidean")
    
    for i in range(n):
        for j in range(i + 1, n):
            if distances[i, j] < threshold:
                s_i = samples[indices[i]]
                s_j = samples[indices[j]]
                # Check if action-relevant variables differ
                ownership_diff = s_i["ball_owner_id"] != s_j["ball_owner_id"]
                teammate_ball_diff = abs(s_i["d_teammate_ball"] - s_j["d_teammate_ball"]) > 0.2
                opponent_ball_diff = abs(s_i["d_opponent_ball"] - s_j["d_opponent_ball"]) > 0.2
                self_goal_diff = abs(s_i["d_self_goal"] - s_j["d_self_goal"]) > 0.2
                
                if ownership_diff or teammate_ball_diff or opponent_ball_diff or self_goal_diff:
                    aliases.append({
                        "s1_idx": int(indices[i]),
                        "s2_idx": int(indices[j]),
                        "obs_distance": round(float(distances[i, j]), 4),
                        "ownership_diff": ownership_diff,
                        "teammate_ball_diff": teammate_ball_diff,
                        "opponent_ball_diff": opponent_ball_diff,
                        "self_goal_diff": self_goal_diff,
                    })
                    if len(aliases) >= max_pairs:
                        break
        if len(aliases) >= max_pairs:
            break
    
    return aliases


def main():
    print("Loading dataset...")
    samples = load_latest_dataset()
    print(f"Loaded {len(samples)} samples")
    
    # Counterfactual tests
    print("\n=== Counterfactual Tests ===")
    pairs = find_counterfactual_pairs(samples, n_pairs=20)
    counterfactual_results = []
    for pair in pairs:
        obs_dist = compute_observation_distance(pair["s1"], pair["s2"])
        counterfactual_results.append({
            "type": pair["type"],
            "expected_diff": pair["expected_diff"],
            "obs_distance": round(obs_dist, 4),
            "s1_seed": pair["s1"]["seed"],
            "s2_seed": pair["s2"]["seed"],
        })
    
    print(f"Tested {len(counterfactual_results)} counterfactual pairs")
    for r in counterfactual_results[:5]:
        print(f"  {r['type']}: obs_dist={r['obs_distance']}")
    
    # State aliasing
    print("\n=== State Aliasing Tests ===")
    aliases = find_state_aliases(samples, threshold=0.1, max_pairs=20)
    print(f"Found {len(aliases)} potential aliases (obs_dist < 0.1 with action-relevant diff)")
    for a in aliases[:5]:
        print(f"  idx={a['s1_idx']},{a['s2_idx']} obs_dist={a['obs_distance']} ownership={a['ownership_diff']} opp_ball={a['opponent_ball_diff']}")
    
    results = {
        "counterfactual_pairs": counterfactual_results,
        "state_aliases": aliases,
        "n_counterfactual_pairs": len(counterfactual_results),
        "n_state_aliases": len(aliases),
    }
    
    out_path = REPORT_DIR / "dobs_counterfactual_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[OUTPUT] Wrote {out_path}")


if __name__ == "__main__":
    main()
