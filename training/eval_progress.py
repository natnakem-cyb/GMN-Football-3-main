"""
GMN-Football-3 -- Persistent Checkpoint Evaluation & Progress Logging
Shared module to evaluate Single-Agent PPO, Multi-Agent IPPO, and Centralized-Critic MAPPO
at milestone increments, appending deterministic evaluation metrics to win_rate_progress.csv.

Schema Migration Guidance (v3.2.0):
- The legacy CSV column `turnover_rate` has been replaced by `non_scoring_episode_rate_pct`
  and `turnovers_conceded_per_ep`.
- `non_scoring_episode_rate_pct` replaces the old misnamed turnover_rate (which was actually
  non_scoring_episodes / num_episodes).
- `turnovers_conceded_per_ep` reports true football turnover metrics sourced from
  FootballMetricsTracker possession-change events (left -> right transitions only).
- `schema_version` is now recorded per evaluation row. Legacy CSV files are preserved
  untouched; new evaluations write with the V2 header.
- When applying this fix across historical logs, rename the old column header to
  `non_scoring_episode_rate_pct` to prevent telemetry distortion; do not convert old values
  as direct turnover metrics.
"""

import hashlib
import json
import os
import sys
import csv
import datetime
import math
import getpass
import platform
from typing import Dict, Any, Optional, List
import numpy as np

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

try:
    from training.football_metrics import FootballMetricsTracker
except ImportError:  # pragma: no cover - defensive fallback
    class FootballMetricsTracker:  # type: ignore
        def __init__(self) -> None:
            self.turnovers_conceded = 0
            self.possession_changes = 0
            self.current_ticks = 0

        def start_episode(self, *args, **kwargs):
            self.turnovers_conceded = 0
            self.possession_changes = 0
            self.current_ticks = 0

        def record_tick(self, *args, **kwargs):
            self.current_ticks += 1

        def end_episode(self, *args, **kwargs):
            return None

DEFAULT_CSV_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "results", "win_rate_progress.csv")
)

SCHEMA_VERSION = "3.2.0"

# Files whose contents meaningfully affect environment dynamics.
# If any of these change, cached evaluations must be considered stale.
ENV_HASH_FILES = [
    os.path.join("src", "engine", "GameEngine.ts"),
    os.path.join("src", "engine", "Physics.ts"),
    os.path.join("src", "engine", "ObservationEncoder.ts"),
    os.path.join("src", "scenarios", "ScenarioRegistry.ts"),
    os.path.join("src", "agents", "RuleBasedAgent.ts"),
]

CSV_FIELDNAMES = [
    "schema_version",
    "evaluation_id",
    "checkpoint_sha256",
    "scenario",
    "algorithm",
    "step",
    "env_version",
    "observation_schema_version",
    "action_schema_version",
    "learning_rate",
    "goal_rate_pct",
    "mean_reward",
    "std_reward",
    "shots_per_ep",
    "non_scoring_episode_rate_pct",
    "turnovers_conceded_per_ep",
    "episodes",
    "deterministic",
    "checkpoint_path",
    "provenance",
    "env_hash",
]


def _extract_owner_team(obs: Optional[np.ndarray]) -> Optional[str]:
    """Decode ball ownership from the observation vector's one-hot slot."""
    if obs is None or len(obs) < 97:
        return None
    ball_owned = obs[94:97]
    if ball_owned[0] > 0.5:
        return None
    if ball_owned[1] > 0.5:
        return "left"
    if ball_owned[2] > 0.5:
        return "right"
    return None


def _extract_ball_pos(obs: Optional[np.ndarray]) -> Dict[str, float]:
    """Extract ball position from the observation vector."""
    if obs is None or len(obs) < 91:
        return {"x": 0.0, "y": 0.0, "z": 0.0}
    return {"x": float(obs[88]), "y": float(obs[89]), "z": float(obs[90])}


def _left_action_indices_single(act_val: int) -> List[int]:
    return [int(act_val)]


def _left_action_indices_multi(action_dict: Dict[str, int]) -> List[int]:
    return [int(v) for k, v in action_dict.items() if k.startswith("left_")]


def _resolve_versioned_csv_path(csv_path: str) -> str:
    """Return a writable CSV path, versioning away from legacy headers if needed."""
    if not os.path.exists(csv_path):
        return csv_path
    try:
        with open(csv_path, "r", newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader, [])
        if "schema_version" in header:
            return csv_path
    except Exception:
        pass
    base, ext = os.path.splitext(csv_path)
    return f"{base}_v2{ext}"




def sha256_file(path: str, chunk_size: int = 1024 * 1024) -> str:
    """
    Computes the lowercase hex SHA256 of a file, reading in chunks.
    Raises FileNotFoundError if the path does not exist.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint file not found: {path}")
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def compute_env_hash() -> str:
    """
    Computes a short SHA-1 hash over the contents of the files that define
    the environment's dynamics (physics, rewards, scenario setup).
    """
    hasher = hashlib.sha1()
    for rel_path in ENV_HASH_FILES:
        abs_path = os.path.join(sys.path[0], rel_path)
        if os.path.exists(abs_path):
            with open(abs_path, "rb") as f:
                hasher.update(f.read())
            hasher.update(rel_path.encode("utf-8"))
    return hasher.hexdigest()[:12]


def load_contract_versions() -> Dict[str, str]:
    """
    Loads version strings from the authoritative TypeScript contract file.
    Falls back to safe defaults if the file cannot be read.
    """
    contract_path = os.path.join(sys.path[0], "src", "engine", "Contract.ts")
    versions = {
        "env_version": "3.1.0",
        "observation_schema_version": "simple115_v3_role",
        "action_schema_version": "discrete19_v1",
    }
    if not os.path.exists(contract_path):
        return versions
    try:
        with open(contract_path, "r", encoding="utf-8") as f:
            text = f.read()
        for key, var_name in [
            ("env_version", "GMN_ENV_VERSION"),
            ("observation_schema_version", "OBSERVATION_SCHEMA_VERSION"),
            ("action_schema_version", "ACTION_SCHEMA_VERSION"),
        ]:
            marker = f"export const {var_name} = '"
            idx = text.find(marker)
            if idx != -1:
                start = idx + len(marker)
                end = text.find("'", start)
                if end != -1:
                    versions[key] = text[start:end]
    except Exception:
        pass
    return versions


def compute_evaluation_identity(
    checkpoint_path: str,
    scenario: str,
    algorithm: str,
    step: int,
    base_seed: int,
    num_episodes: int,
    deterministic: bool,
    env_hash: str,
) -> Dict[str, Any]:
    """
    Builds a canonical evaluation identity dict and derives a stable evaluation_id.
    """
    contract = load_contract_versions()
    checkpoint_sha256 = sha256_file(checkpoint_path)
    identity = {
        "checkpoint_sha256": checkpoint_sha256,
        "scenario": scenario,
        "algorithm": algorithm.upper(),
        "env_version": contract["env_version"],
        "observation_schema_version": contract["observation_schema_version"],
        "action_schema_version": contract["action_schema_version"],
        "step": step,
        "base_seed": base_seed,
        "num_episodes": num_episodes,
        "deterministic": deterministic,
        "env_hash": env_hash,
    }
    canonical = json.dumps(identity, sort_keys=True, separators=(",", ":"))
    evaluation_id = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]
    return {
        "evaluation_id": evaluation_id,
        "checkpoint_sha256": checkpoint_sha256,
        "scenario": scenario,
        "algorithm": algorithm.upper(),
        "env_version": contract["env_version"],
        "observation_schema_version": contract["observation_schema_version"],
        "action_schema_version": contract["action_schema_version"],
        "step": step,
        "base_seed": base_seed,
        "num_episodes": num_episodes,
        "deterministic": deterministic,
        "env_hash": env_hash,
    }


def check_existing_evaluation(
    csv_path: str, evaluation_id: str
) -> Optional[Dict[str, Any]]:
    """
    Checks whether an evaluation with the exact evaluation_id is already present in the CSV.
    Returns the parsed row dict if found, else None.

    Legacy CSV entries without an evaluation_id column are treated as non-matching.
    """
    if not os.path.exists(csv_path):
        return None

    try:
        with open(csv_path, mode="r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if not reader.fieldnames or "evaluation_id" not in reader.fieldnames:
                # Legacy CSV without evaluation_id column - no exact matches possible
                return None
            for row in reader:
                if row.get("evaluation_id") == evaluation_id:
                    return row
    except Exception as e:
        print(f"[eval_progress] Warning: failed reading CSV {csv_path}: {e}")
    return None


def append_progress_row(csv_path: str, row_dict: Dict[str, Any]) -> None:
    """
    Appends a formatted evaluation record to the target CSV file, creating directories
    and header as needed. Preserves legacy CSV files by versioning new writes when the
    existing file does not contain the current schema_version column.
    """
    target_path = _resolve_versioned_csv_path(csv_path)
    os.makedirs(os.path.dirname(os.path.abspath(target_path)), exist_ok=True)
    file_exists = os.path.exists(target_path) and os.path.getsize(target_path) > 0

    with open(target_path, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES)
        if not file_exists:
            writer.writeheader()
        writer.writerow({k: row_dict.get(k, "") for k in CSV_FIELDNAMES})


def evaluate_single_agent_ppo(
    checkpoint_path: str,
    scenario: str,
    num_episodes: int = 50,
    deterministic: bool = True,
    base_seed: int = 500000,
) -> Dict[str, float]:
    from training.gmn_gym import GMNFootballEnv
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    raw_env = GMNFootballEnv(scenario=scenario, port=5050, use_ws=True)
    vec_env = DummyVecEnv([lambda: raw_env])

    vec_norm_path = checkpoint_path.replace(".zip", "_vecnormalize.pkl")
    if os.path.exists(vec_norm_path):
        vec_env = VecNormalize.load(vec_norm_path, vec_env)
        vec_env.training = False
        vec_env.norm_reward = False

    try:
        model = PPO.load(checkpoint_path, env=vec_env)
    except Exception:
        model = PPO.load(checkpoint_path)

    rewards = []
    goals = 0
    shots = 0
    turnovers_conceded_total = 0.0
    is_rondo = scenario == "academy_rondo_4v1"

    try:
        for ep in range(num_episodes):
            seed = base_seed + ep * 1009
            obs = vec_env.reset()
            tracker = FootballMetricsTracker()
            tracker.start_episode(scenario, seed, _extract_ball_pos(obs))
            ep_rew = 0.0
            steps = 0
            done = False
            ep_shot = False
            last_info = {}

            while not done and steps < 600:
                action, _ = model.predict(obs, deterministic=deterministic)
                act_val = int(action[0]) if isinstance(action, (list, np.ndarray)) else int(action)
                if not is_rondo and act_val == 12:  # Shot action
                    shots += 1
                    ep_shot = True

                obs, reward_arr, done_arr, info_list = vec_env.step(action)
                reward = float(reward_arr[0])
                done = bool(done_arr[0])
                last_info = info_list[0] if info_list else {}
                ep_rew += reward
                steps += 1

                owner_team = _extract_owner_team(obs)
                ball_pos = _extract_ball_pos(obs)
                tracker.record_tick(
                    _left_action_indices_single(act_val),
                    reward,
                    ball_pos,
                    owner_team,
                )

            tracker.end_episode(
                last_info.get("score", {"left": 0, "right": 0}),
                {},
                _extract_ball_pos(obs),
            )
            ep_metrics = tracker.episodes[-1] if tracker.episodes else None
            turnovers_conceded_total += ep_metrics.turnovers_conceded if ep_metrics else 0.0

            rewards.append(ep_rew)
            score_left = last_info.get("score", {}).get("left", 0)
            event = last_info.get("event", {})
            is_goal = score_left > 0 or (isinstance(event, dict) and event.get("type") == "goal")

            if is_goal:
                goals += 1
    finally:
        vec_env.close()

    mean_rew = float(np.mean(rewards)) if rewards else 0.0
    std_rew = float(np.std(rewards)) if rewards else 0.0
    goal_rate_pct = (goals / max(1, num_episodes)) * 100.0
    shots_per_ep = shots / max(1, num_episodes)
    non_scoring_episode_rate_pct = ((num_episodes - goals) / max(1, num_episodes)) * 100.0
    turnovers_conceded_per_ep = turnovers_conceded_total / max(1, num_episodes)

    if is_rondo:
        possession_retention_time = float("nan")
        completed_pass_chains = float("nan")
    else:
        possession_retention_time = float("nan")
        completed_pass_chains = float("nan")

    return {
        "goal_rate_pct": goal_rate_pct,
        "mean_reward": mean_rew,
        "std_reward": std_rew,
        "shots_per_ep": shots_per_ep,
        "non_scoring_episode_rate_pct": non_scoring_episode_rate_pct,
        "turnovers_conceded_per_ep": turnovers_conceded_per_ep,
        "possession_retention_time": possession_retention_time,
        "completed_pass_chains": completed_pass_chains,
    }


def evaluate_multi_agent_ippo(
    checkpoint_path: str,
    scenario: str,
    num_episodes: int = 50,
    deterministic: bool = True,
    base_seed: int = 500000,
) -> Dict[str, float]:
    from training.gmn_pettingzoo import GMNMultiAgentEnv
    from stable_baselines3 import PPO

    model = PPO.load(checkpoint_path)
    env = GMNMultiAgentEnv(scenario=scenario, auto_start_bridge=True)

    rewards = []
    goals = 0
    shots = 0
    turnovers_conceded_total = 0.0
    is_rondo = scenario == "academy_rondo_4v1"

    try:
        for ep in range(num_episodes):
            seed = base_seed + ep * 1009
            obs_dict, _ = env.reset(seed=seed)
            tracker = FootballMetricsTracker()
            sample_obs = next(iter(obs_dict.values())) if obs_dict else None
            tracker.start_episode(scenario, seed, _extract_ball_pos(sample_obs))
            ep_rew = 0.0
            steps = 0
            done = False
            last_info = {}

            while not done and steps < 600:
                actions = {}
                for agent_id in env.agents:
                    obs = obs_dict[agent_id]
                    act, _ = model.predict(obs, deterministic=deterministic)
                    act_int = int(act)
                    actions[agent_id] = act_int
                    if not is_rondo and act_int == 12:
                        shots += 1

                obs_dict, rews, terms, truncs, infos = env.step(actions)
                steps += 1

                if env.possible_agents and env.possible_agents[0] in rews:
                    ep_rew += float(rews[env.possible_agents[0]])

                term = any(terms.values()) if terms else False
                trunc = any(truncs.values()) if truncs else False
                done = term or trunc or not env.agents

                if infos:
                    for inf in infos.values():
                        last_info = inf
                        break

                sample_obs = next(iter(obs_dict.values())) if obs_dict else None
                owner_team = _extract_owner_team(sample_obs)
                ball_pos = _extract_ball_pos(sample_obs)
                tracker.record_tick(
                    _left_action_indices_multi(actions),
                    float(rews.get(env.possible_agents[0], 0.0)) if env.possible_agents else 0.0,
                    ball_pos,
                    owner_team,
                )

            tracker.end_episode(
                last_info.get("score", {"left": 0, "right": 0}),
                {},
                _extract_ball_pos(sample_obs),
            )
            ep_metrics = tracker.episodes[-1] if tracker.episodes else None
            turnovers_conceded_total += ep_metrics.turnovers_conceded if ep_metrics else 0.0

            rewards.append(ep_rew)
            if not is_rondo:
                score_left = last_info.get("score", {}).get("left", 0)
                event = last_info.get("event", {})
                is_goal = score_left > 0 or (isinstance(event, dict) and event.get("type") == "goal")

                if is_goal:
                    goals += 1
    finally:
        env.close()

    mean_rew = float(np.mean(rewards)) if rewards else 0.0
    std_rew = float(np.std(rewards)) if rewards else 0.0
    goal_rate_pct = (goals / max(1, num_episodes)) * 100.0
    shots_per_ep = shots / max(1, num_episodes)
    non_scoring_episode_rate_pct = ((num_episodes - goals) / max(1, num_episodes)) * 100.0
    turnovers_conceded_per_ep = turnovers_conceded_total / max(1, num_episodes)

    if is_rondo:
        possession_retention_time = float("nan")
        completed_pass_chains = float("nan")
    else:
        possession_retention_time = float("nan")
        completed_pass_chains = float("nan")

    return {
        "goal_rate_pct": goal_rate_pct,
        "mean_reward": mean_rew,
        "std_reward": std_rew,
        "shots_per_ep": shots_per_ep,
        "non_scoring_episode_rate_pct": non_scoring_episode_rate_pct,
        "turnovers_conceded_per_ep": turnovers_conceded_per_ep,
        "possession_retention_time": possession_retention_time,
        "completed_pass_chains": completed_pass_chains,
    }


def evaluate_multi_agent_mappo(
    checkpoint_path: str,
    scenario: str,
    num_episodes: int = 50,
    deterministic: bool = True,
    base_seed: int = 500000,
) -> Dict[str, float]:
    import torch
    from training.gmn_pettingzoo import GMNMultiAgentEnv
    from training.mappo_networks import SharedActor

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    obs_dim = checkpoint.get("obs_dim", 127 if "actor" in checkpoint and checkpoint["actor"]["net.0.weight"].shape[1] == 127 else (checkpoint["actor"]["net.0.weight"].shape[1] if "actor" in checkpoint else 127))
    action_dim = checkpoint.get("action_dim", 19)

    actor = SharedActor(obs_dim=obs_dim, action_dim=action_dim, hidden=64)
    actor.load_state_dict(checkpoint["actor"])
    actor.eval()

    env = GMNMultiAgentEnv(scenario=scenario, auto_start_bridge=True)
    controllable_agents = list(env.possible_agents)

    rewards = []
    goals = 0
    shots = 0
    turnovers_conceded_total = 0.0
    is_rondo = scenario == "academy_rondo_4v1"

    try:
        for ep in range(num_episodes):
            seed = base_seed + ep * 1009
            obs_dict, _ = env.reset(seed=seed)
            tracker = FootballMetricsTracker()
            sample_obs = next(iter(obs_dict.values())) if obs_dict else None
            tracker.start_episode(scenario, seed, _extract_ball_pos(sample_obs))
            ep_rew = 0.0
            steps = 0
            done = False
            last_info = {}

            while not done and steps < 600:
                current_agents = list(env.agents if env.agents else controllable_agents)
                local_obs = np.stack([obs_dict[a] for a in current_agents], axis=0).astype(np.float32)

                with torch.no_grad():
                    dist = actor(torch.from_numpy(local_obs).float())
                    if deterministic:
                        actions = dist.logits.argmax(dim=-1)
                    else:
                        actions = dist.sample()

                action_dict = {}
                for i, a in enumerate(current_agents):
                    act_int = int(actions[i].item())
                    action_dict[a] = act_int
                    if not is_rondo and act_int == 12:
                        shots += 1

                obs_dict, rews, terms, truncs, infos = env.step(action_dict)
                steps += 1

                shared_rew = float(rews[current_agents[0]]) if current_agents and current_agents[0] in rews else 0.0
                ep_rew += shared_rew

                term = any(terms.values()) if terms else False
                trunc = any(truncs.values()) if truncs else False
                done = term or trunc or not env.agents

                if infos:
                    for inf in infos.values():
                        last_info = inf
                        break

                sample_obs = next(iter(obs_dict.values())) if obs_dict else None
                owner_team = _extract_owner_team(sample_obs)
                ball_pos = _extract_ball_pos(sample_obs)
                tracker.record_tick(
                    _left_action_indices_multi(action_dict),
                    shared_rew,
                    ball_pos,
                    owner_team,
                )

            tracker.end_episode(
                last_info.get("score", {"left": 0, "right": 0}),
                {},
                _extract_ball_pos(sample_obs),
            )
            ep_metrics = tracker.episodes[-1] if tracker.episodes else None
            turnovers_conceded_total += ep_metrics.turnovers_conceded if ep_metrics else 0.0

            rewards.append(ep_rew)
            if not is_rondo:
                score_left = last_info.get("score", {}).get("left", 0)
                event = last_info.get("event", {})
                is_goal = score_left > 0 or (isinstance(event, dict) and event.get("type") == "goal")

                if is_goal:
                    goals += 1
    finally:
        env.close()

    mean_rew = float(np.mean(rewards)) if rewards else 0.0
    std_rew = float(np.std(rewards)) if rewards else 0.0
    goal_rate_pct = (goals / max(1, num_episodes)) * 100.0
    shots_per_ep = shots / max(1, num_episodes)
    non_scoring_episode_rate_pct = ((num_episodes - goals) / max(1, num_episodes)) * 100.0
    turnovers_conceded_per_ep = turnovers_conceded_total / max(1, num_episodes)

    if is_rondo:
        possession_retention_time = float("nan")
        completed_pass_chains = float("nan")
    else:
        possession_retention_time = float("nan")
        completed_pass_chains = float("nan")

    return {
        "goal_rate_pct": goal_rate_pct,
        "mean_reward": mean_rew,
        "std_reward": std_rew,
        "shots_per_ep": shots_per_ep,
        "non_scoring_episode_rate_pct": non_scoring_episode_rate_pct,
        "turnovers_conceded_per_ep": turnovers_conceded_per_ep,
        "possession_retention_time": possession_retention_time,
        "completed_pass_chains": completed_pass_chains,
    }


def evaluate_checkpoint_progress(
    checkpoint_path: str,
    scenario: str,
    algorithm: str,
    step: int,
    learning_rate: float = 3e-4,
    num_episodes: int = 50,
    deterministic: bool = True,
    base_seed: int = 500000,
    csv_path: str = DEFAULT_CSV_PATH,
    force_reeval: bool = False,
) -> Dict[str, Any]:
    """
    Loads checkpoint, runs deterministic evaluation rollout, and appends row to CSV.
    Cache key is a canonical evaluation_id derived from checkpoint SHA256, scenario,
    algorithm, step, env_hash, and evaluation config. Skips re-evaluating if an exact
    matching entry exists unless force_reeval=True.
    """
    algo_upper = algorithm.upper()
    env_hash = compute_env_hash()
    identity = compute_evaluation_identity(
        checkpoint_path=checkpoint_path,
        scenario=scenario,
        algorithm=algo_upper,
        step=step,
        base_seed=base_seed,
        num_episodes=num_episodes,
        deterministic=deterministic,
        env_hash=env_hash,
    )
    evaluation_id = identity["evaluation_id"]
    checkpoint_sha256 = identity["checkpoint_sha256"]

    if not force_reeval:
        existing = check_existing_evaluation(csv_path, evaluation_id)
        if existing is not None:
            print(
                f"[eval_progress] Cache hit for evaluation_id={evaluation_id} "
                f"(scenario={scenario}, algorithm={algo_upper}, step={step}, "
                f"checkpoint_sha256={checkpoint_sha256}). "
                f"Goal Rate: {float(existing.get('goal_rate_pct', 0.0)):.1f}%. Skipping re-evaluation."
            )
            return existing

    print(
        f"\n[eval_progress] >>> Evaluating Checkpoint Milestone: {algo_upper} | Scenario: {scenario} | "
        f"Step: {step:,} | LR: {learning_rate:g} | Episodes: {num_episodes} | "
        f"checkpoint_sha256={checkpoint_sha256} <<<"
    )

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    if algo_upper == "PPO":
        eval_metrics = evaluate_single_agent_ppo(
            checkpoint_path=checkpoint_path,
            scenario=scenario,
            num_episodes=num_episodes,
            deterministic=deterministic,
            base_seed=base_seed,
        )
    elif algo_upper == "IPPO":
        eval_metrics = evaluate_multi_agent_ippo(
            checkpoint_path=checkpoint_path,
            scenario=scenario,
            num_episodes=num_episodes,
            deterministic=deterministic,
            base_seed=base_seed,
        )
    elif algo_upper == "MAPPO":
        eval_metrics = evaluate_multi_agent_mappo(
            checkpoint_path=checkpoint_path,
            scenario=scenario,
            num_episodes=num_episodes,
            deterministic=deterministic,
            base_seed=base_seed,
        )
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}. Expected PPO, IPPO, or MAPPO.")

    provenance_host = platform.node()
    provenance_user = getpass.getuser()
    provenance_date = datetime.datetime.now().isoformat()
    provenance_cmd = " ".join(sys.argv)
    provenance_str = f"host={provenance_host}|user={provenance_user}|date={provenance_date}|cmd={provenance_cmd}"

    row = {
        "schema_version": SCHEMA_VERSION,
        "evaluation_id": evaluation_id,
        "checkpoint_sha256": checkpoint_sha256,
        "scenario": scenario,
        "algorithm": algo_upper,
        "env_version": identity["env_version"],
        "observation_schema_version": identity["observation_schema_version"],
        "action_schema_version": identity["action_schema_version"],
        "step": step,
        "learning_rate": f"{learning_rate:g}",
        "goal_rate_pct": f"{eval_metrics['goal_rate_pct']:.2f}",
        "mean_reward": f"{eval_metrics['mean_reward']:.4f}",
        "std_reward": f"{eval_metrics['std_reward']:.4f}",
        "shots_per_ep": f"{eval_metrics['shots_per_ep']:.2f}",
        "non_scoring_episode_rate_pct": f"{eval_metrics['non_scoring_episode_rate_pct']:.2f}",
        "turnovers_conceded_per_ep": f"{eval_metrics['turnovers_conceded_per_ep']:.2f}",
        "episodes": num_episodes,
        "deterministic": deterministic,
        "checkpoint_path": checkpoint_path,
        "provenance": provenance_str,
        "env_hash": env_hash,
    }

    append_progress_row(csv_path, row)
    print(
        f"[eval_progress] [OK] Milestone logged -> Goal Rate: {eval_metrics['goal_rate_pct']:.1f}% | "
        f"Mean Reward: {eval_metrics['mean_reward']:+.4f} | Shots/Ep: {eval_metrics['shots_per_ep']:.2f} | "
        f"Non-Scoring Episode Rate: {eval_metrics['non_scoring_episode_rate_pct']:.1f}% | "
        f"Turnovers Conceded/Ep: {eval_metrics['turnovers_conceded_per_ep']:.2f} | "
        f"evaluation_id={evaluation_id} | checkpoint_sha256={checkpoint_sha256} | "
        f"CSV: {csv_path}\n"
    )
    return row


def persist_trend_snapshots(
    snapshots: list,
    algorithm: str,
    scenario: str,
    output_dir: str = os.path.join(os.path.dirname(__file__), "results"),
    seed: int = None,
) -> str:
    """
    Persists in-training trend snapshots to training/results/trend_<algorithm>_<scenario>.csv.
    When a seed is provided, the filename is suffixed with _seed<seed> to avoid
    write-mode races between concurrent training runs.
    """
    os.makedirs(output_dir, exist_ok=True)
    algo_lower = algorithm.lower()
    seed_suffix = f"_seed{seed}" if seed is not None else ""
    filename = f"trend_{algo_lower}_{scenario}{seed_suffix}.csv"
    csv_path = os.path.join(output_dir, filename)

    with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "episodes", "mean_reward", "goal_rate_pct"])
        for item in snapshots:
            step, num_eps, mean_rew, goal_pct = item
            writer.writerow([step, num_eps, f"{mean_rew:.4f}", f"{goal_pct:.2f}"])

    print(f"   [OK] Trend snapshots persisted to: {csv_path}")
    return csv_path
