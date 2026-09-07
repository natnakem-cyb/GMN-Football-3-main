"""
GMN-Football-3 — Opponent Pool & Self-Play Support

Manages a pool of opponents used by the training environment wrappers for the
non-learning (right) team. Two kinds of opponents are supported:

1. **Rule-based difficulty levels** ('easy' | 'medium' | 'hard' | 'master').
   These are executable today: the pool selects a level each episode and the
   wrapper applies it to the bridge via ``POST /opponent``.
2. **Learned policy snapshots** (path to a saved actor checkpoint). Snapshotting
   works today (periodic ``torch.save`` of the training policy); *playing*
   against a learned snapshot requires the bridge to load and run that policy
   for the right team, which is future work (tracked in training/README.md).

Selection strategies:
- ``uniform``   — sample uniformly at random each episode (default)
- ``cyclic``    — cycle through the pool in a fixed order
- ``elo``       — softmax weighting by stored Elo ratings (updated via ``report_result``)
"""

import os
import random
import time
from typing import Any, Dict, List, Optional

DIFFICULTY_LEVELS = ("easy", "medium", "hard", "master")


class OpponentPool:
    """Pool of opponent specifications with pluggable selection strategy."""

    def __init__(
        self,
        difficulties: Optional[List[str]] = None,
        strategy: str = "uniform",
        seed: Optional[int] = None,
        snapshot_dir: str = "training/models/opponent_pool",
        snapshot_every_steps: int = 50_000,
        max_snapshots: int = 5,
    ):
        assert strategy in ("uniform", "cyclic", "elo"), f"Unknown strategy: {strategy}"
        self.strategy = strategy
        self.rng = random.Random(seed)
        self.snapshot_dir = snapshot_dir
        self.snapshot_every_steps = snapshot_every_steps
        self.max_snapshots = max_snapshots

        # entries: {"kind": "difficulty", "difficulty": str, "elo": 1000.0, "games": 0}
        #          {"kind": "snapshot", "path": str, "elo": 1000.0, "games": 0}
        self.entries: List[Dict[str, Any]] = []
        for d in (difficulties or list(DIFFICULTY_LEVELS)):
            self.add_difficulty(d)

        self._cyclic_idx = 0
        self._last_snapshot_step = -self.snapshot_every_steps
        os.makedirs(self.snapshot_dir, exist_ok=True)

    def add_difficulty(self, difficulty: str) -> None:
        assert difficulty in DIFFICULTY_LEVELS, f"Invalid difficulty: {difficulty}"
        self.entries.append({"kind": "difficulty", "difficulty": difficulty, "elo": 1000.0, "games": 0})

    def add_snapshot(self, path: str) -> None:
        self.entries.append({"kind": "snapshot", "path": path, "elo": 1000.0, "games": 0})

    def sample(self) -> Dict[str, Any]:
        """Select the opponent spec to use for the next episode."""
        if not self.entries:
            raise RuntimeError("OpponentPool is empty")
        if self.strategy == "cyclic":
            entry = self.entries[self._cyclic_idx % len(self.entries)]
            self._cyclic_idx += 1
            return entry
        if self.strategy == "elo":
            # Softmax weighting over stored Elo ratings (temperature 200).
            import math
            elos = [e["elo"] for e in self.entries]
            m = max(elos)
            weights = [math.exp((e - m) / 200.0) for e in elos]
            return self.rng.choices(self.entries, weights=weights, k=1)[0]
        return self.rng.choice(self.entries)

    def apply(self, entry: Dict[str, Any], env: Any) -> bool:
        """
        Apply the selected opponent to an environment. Returns True if the
        opponent was actually applied (rule-based difficulties only, today).
        """
        if entry["kind"] == "difficulty":
            env.set_opponent_difficulty(entry["difficulty"])
            return True
        # Learned snapshots cannot yet be executed by the Node bridge.
        return False

    def report_result(self, entry: Dict[str, Any], score: float, k: float = 16.0) -> None:
        """Update Elo after an episode. score: 1.0 win / 0.5 draw / 0.0 loss."""
        expected = 1.0 / (1.0 + 10 ** ((1200.0 - entry["elo"]) / 400.0))
        entry["elo"] += k * (score - expected)
        entry["games"] += 1

    def maybe_snapshot(self, actor: Any, step: int) -> Optional[str]:
        """Periodically save the current training policy as a pool snapshot."""
        if step - self._last_snapshot_step < self.snapshot_every_steps:
            return None
        self._last_snapshot_step = step
        try:
            import torch  # local import: only needed when snapshotting

            path = os.path.join(self.snapshot_dir, f"actor_{int(time.time())}_{step}.pt")
            state = actor.state_dict() if hasattr(actor, "state_dict") else actor
            torch.save(state, path)
            self.add_snapshot(path)
            # Bound the pool: drop the oldest snapshot entries beyond the limit.
            snaps = [e for e in self.entries if e["kind"] == "snapshot"]
            for old in snaps[: max(0, len(snaps) - self.max_snapshots)]:
                self.entries.remove(old)
                try:
                    os.remove(old["path"])
                except OSError:
                    pass
            return path
        except Exception as e:  # snapshotting must never crash training
            print(f"[OpponentPool] Snapshot failed: {e}")
            return None

    def describe(self) -> str:
        parts = [f"{e['kind']}:{e.get('difficulty', e.get('path', '?'))}" for e in self.entries]
        return f"OpponentPool(strategy={self.strategy}, entries=[{', '.join(parts)}])"
