"""D-Obs: Observation sufficiency probes for academy_3_vs_1_with_keeper.

This script tests whether the 127-dim simple115_v3_role observation
contains sufficient information to recover critical football state variables.

Methodology:
- Direct recovery: compute ground-truth quantities from observation
  positions directly (tests information presence)
- Classifier probes: use k-NN (non-linear) to test recoverability
- Counterfactual tests: paired states with controlled semantic changes
"""

import json
import sys
import os
import numpy as np
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

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
    
    # Role one-hot: indices 115-126
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


def prepare_arrays(samples: list) -> tuple:
    X = np.array([s["raw_observation"] for s in samples], dtype=np.float32)
    
    ownership_map = {None: 0, "left": 1, "right": 2}
    y_owner = np.array([ownership_map.get(s["ball_owner_id"], 0) for s in samples])
    y_d_self_ball = np.array([s["d_self_ball"] for s in samples])
    y_d_teammate_ball = np.array([s["d_teammate_ball"] for s in samples])
    y_d_opponent_ball = np.array([s["d_opponent_ball"] for s in samples])
    y_d_self_goal = np.array([s["d_self_goal"] for s in samples])
    y_d_teammate_goal = np.array([s["d_teammate_goal"] for s in samples])
    
    # Ball displacement direction thresholds
    y_ball_dir = np.array([
        0 if s["ball_delta_x"] < -0.001 else (2 if s["ball_delta_x"] > 0.001 else 1)
        for s in samples
    ])
    
    role_vocab = ["GK", "CB", "LB", "RB", "CDM", "CM", "LM", "RM", "LW", "RW", "CAM", "ST"]
    role_to_idx = {r: i for i, r in enumerate(role_vocab)}
    y_role = np.array([role_to_idx.get(s["self_role"], 0) for s in samples])
    
    return X, y_owner, y_d_self_ball, y_d_teammate_ball, y_d_opponent_ball, y_d_self_goal, y_d_teammate_goal, y_ball_dir, y_role


def classification_probe(name: str, X_train, X_test, y_train, y_test) -> dict:
    """Run a classification probe with k-NN and Random Forest."""
    if len(np.unique(y_train)) < 2:
        return {
            "name": name,
            "status": "NOT_TESTABLE",
            "reason": "Only one class in training data",
        }
    
    # k-NN probe
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)
    y_pred_knn = knn.predict(X_test)
    knn_acc = accuracy_score(y_test, y_pred_knn)
    knn_f1 = f1_score(y_test, y_pred_knn, average="macro", zero_division=0)
    
    # Random Forest probe
    rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced")
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    rf_acc = accuracy_score(y_test, y_pred_rf)
    rf_f1 = f1_score(y_test, y_pred_rf, average="macro", zero_division=0)
    
    classes = sorted(np.unique(y_test).tolist())
    class_counts = {int(c): int((y_test == c).sum()) for c in classes}
    cm = confusion_matrix(y_test, y_pred_rf).tolist()
    
    return {
        "name": name,
        "status": "TESTED",
        "knn_accuracy": round(float(knn_acc), 4),
        "knn_macro_f1": round(float(knn_f1), 4),
        "rf_accuracy": round(float(rf_acc), 4),
        "rf_macro_f1": round(float(rf_f1), 4),
        "confusion_matrix_rf": cm,
        "classes": classes,
        "class_counts": class_counts,
        "n_train": len(y_train),
        "n_test": len(y_test),
    }


def direct_recovery_probe(samples: list) -> dict:
    """Test exact recoverability of distances from observation positions."""
    valid_samples = []
    for s in samples:
        obs = np.array(s["raw_observation"])
        parsed = extract_positions(obs)
        active_pos = parsed["left_positions"][parsed["active_idx"]]
        if active_pos[0] == -1.0 or active_pos[1] == -1.0:
            continue
        if parsed["ball_pos"][0] == -1.0:
            continue
        valid_samples.append((s, obs, parsed))
    
    errors = defaultdict(list)
    ownership_correct = []
    
    for s, obs, parsed in valid_samples:
        active_pos = parsed["left_positions"][parsed["active_idx"]]
        ball_pos_xy = (parsed["ball_pos"][0], parsed["ball_pos"][1])
        goal_pos = (1.0, 0.0)
        
        # Self-ball
        computed = np.hypot(active_pos[0] - ball_pos_xy[0], active_pos[1] - ball_pos_xy[1])
        errors["self_ball"].append(abs(computed - s["d_self_ball"]))
        
        # Self-goal
        computed = np.hypot(active_pos[0] - goal_pos[0], active_pos[1] - goal_pos[1])
        errors["self_goal"].append(abs(computed - s["d_self_goal"]))
        
        # Teammate-ball (nearest)
        teammates = [i for i in range(11) if i != parsed["active_idx"] and parsed["left_positions"][i][0] != -1.0]
        if teammates:
            min_d = min(np.hypot(parsed["left_positions"][i][0] - ball_pos_xy[0], parsed["left_positions"][i][1] - ball_pos_xy[1]) for i in teammates)
            errors["teammate_ball"].append(abs(min_d - s["d_teammate_ball"]))
        
        # Opponent-ball (nearest)
        opponents = [i for i in range(11) if parsed["right_positions"][i][0] != -1.0]
        if opponents:
            min_d = min(np.hypot(parsed["right_positions"][i][0] - ball_pos_xy[0], parsed["right_positions"][i][1] - ball_pos_xy[1]) for i in opponents)
            errors["opponent_ball"].append(abs(min_d - s["d_opponent_ball"]))
        
        # Teammate-goal (nearest)
        if teammates:
            min_d = min(np.hypot(parsed["left_positions"][i][0] - goal_pos[0], parsed["left_positions"][i][1] - goal_pos[1]) for i in teammates)
            errors["teammate_goal"].append(abs(min_d - s["d_teammate_goal"]))
        
        # Ownership
        expected = 0
        if s["ball_owner_id"] is None:
            expected = 0
        elif s["ball_owner_id"].startswith("left"):
            expected = 1
        elif s["ball_owner_id"].startswith("right"):
            expected = 2
        ownership_correct.append(1 if parsed["ownership"] == expected else 0)
    
    result = {
        "self_ball_exact_recovery_mae": round(float(np.mean(errors["self_ball"])), 6),
        "self_goal_exact_recovery_mae": round(float(np.mean(errors["self_goal"])), 6),
        "teammate_ball_exact_recovery_mae": round(float(np.mean(errors["teammate_ball"])) if errors["teammate_ball"] else None, 6),
        "opponent_ball_exact_recovery_mae": round(float(np.mean(errors["opponent_ball"])) if errors["opponent_ball"] else None, 6),
        "teammate_goal_exact_recovery_mae": round(float(np.mean(errors["teammate_goal"])) if errors["teammate_goal"] else None, 6),
        "ownership_exact_recovery_accuracy": round(float(np.mean(ownership_correct)), 6),
        "n_valid": len(valid_samples),
    }
    return result


def main():
    print("Loading dataset...")
    samples = load_latest_dataset()
    print(f"Loaded {len(samples)} samples")
    
    X, y_owner, y_d_self_ball, y_d_teammate_ball, y_d_opponent_ball, y_d_self_goal, y_d_teammate_goal, y_ball_dir, y_role = prepare_arrays(samples)
    
    # Direct recovery probe
    print("\n=== Direct Recovery Probe ===")
    direct = direct_recovery_probe(samples)
    for k, v in direct.items():
        print(f"  {k}: {v}")
    
    # Use stratified splits where possible
    try:
        X_train, X_test, y_owner_train, y_owner_test = train_test_split(X, y_owner, test_size=0.3, random_state=42, stratify=y_owner)
    except ValueError:
        X_train, X_test, y_owner_train, y_owner_test = train_test_split(X, y_owner, test_size=0.3, random_state=42)
    
    _, _, y_ball_dir_train, y_ball_dir_test = train_test_split(X, y_ball_dir, test_size=0.3, random_state=42)
    _, _, y_role_train, y_role_test = train_test_split(X, y_role, test_size=0.3, random_state=42)
    
    results = {"direct_recovery": direct}
    
    print("\n=== Ball Displacement Direction Probe ===")
    results["ball_direction"] = classification_probe(
        "Ball displacement direction",
        X_train, X_test, y_ball_dir_train, y_ball_dir_test
    )
    print(results["ball_direction"])
    
    print("\n=== Role Probe ===")
    results["role"] = classification_probe(
        "Self role",
        X_train, X_test, y_role_train, y_role_test
    )
    print(results["role"])
    
    # Save results
    out_path = REPORT_DIR / "dobs_probe_results_v2.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[OUTPUT] Wrote {out_path}")


if __name__ == "__main__":
    main()
