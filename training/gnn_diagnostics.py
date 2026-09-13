"""
GMN-Football-3 — Diagnostic Probes (Phase 4)

Lightweight supervised probes attached to frozen GNN embeddings.
Each probe tests whether a specific class of information is preserved
in the learned representation.

Probes:
- SpatialGeometryProbe: player-player/ball/goal distances
- DirectionProbe: bearing to goal, goal-mouth angle
- PressureProbe: nearest-opponent distance
- FormationProbe: formation identity, slot role, line, lane
- GoalGeometryProbe: shot distance, shot angle
- PassingProbe: teammate distance, direction, relative velocity
- ScenarioProbe: scenario identity from graph + zScenario
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Base Probe
# ---------------------------------------------------------------------------

class BaseProbe(nn.Module):
    """Base class for diagnostic probes."""

    def __init__(self, input_dim: int, output_dim: int, task_type: str = "regression"):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.task_type = task_type  # 'regression' or 'classification'

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Spatial Geometry Probe
# ---------------------------------------------------------------------------

class SpatialGeometryProbe(BaseProbe):
    """
    Probes whether embeddings preserve spatial distances.
    Target: player-player, player-ball, player-goal distances.
    """

    def __init__(self, input_dim: int, num_targets: int = 3):
        super().__init__(input_dim, num_targets, task_type="regression")
        self.net = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.Linear(input_dim // 2, num_targets),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# Direction / Angle Probe
# ---------------------------------------------------------------------------

class DirectionProbe(BaseProbe):
    """
    Probes whether embeddings preserve directional information.
    Target: bearing to goal, goal-mouth opening angle.
    """

    def __init__(self, input_dim: int, num_targets: int = 2):
        super().__init__(input_dim, num_targets, task_type="regression")
        self.net = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.Linear(input_dim // 2, num_targets),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# Pressure Probe
# ---------------------------------------------------------------------------

class PressureProbe(BaseProbe):
    """
    Probes whether embeddings preserve opponent proximity information.
    Target: nearest-opponent distance.
    """

    def __init__(self, input_dim: int, num_targets: int = 1):
        super().__init__(input_dim, num_targets, task_type="regression")
        self.net = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.Linear(input_dim // 2, num_targets),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# Formation Probe
# ---------------------------------------------------------------------------

class FormationProbe(BaseProbe):
    """
    Probes whether embeddings preserve formation/tactical information.
    Targets: formation identity, slot role, line, lane.
    """

    def __init__(self, input_dim: int, num_formations: int = 1, num_roles: int = 12, num_lines: int = 4, num_lanes: int = 3):
        super().__init__(input_dim, num_formations + num_roles + num_lines + num_lanes, task_type="classification")
        self.num_formations = num_formations
        self.num_roles = num_roles
        self.num_lines = num_lines
        self.num_lanes = num_lanes
        self.net = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.Linear(input_dim // 2, self.output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# Goal Geometry Probe
# ---------------------------------------------------------------------------

class GoalGeometryProbe(BaseProbe):
    """
    Probes whether embeddings preserve goal geometry.
    Target: shot distance, shot angle.
    """

    def __init__(self, input_dim: int, num_targets: int = 2):
        super().__init__(input_dim, num_targets, task_type="regression")
        self.net = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.Linear(input_dim // 2, num_targets),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# Passing Relationship Probe
# ---------------------------------------------------------------------------

class PassingProbe(BaseProbe):
    """
    Probes whether embeddings preserve passing-related geometry.
    Target: teammate distance, direction, relative velocity.
    """

    def __init__(self, input_dim: int, num_targets: int = 4):
        super().__init__(input_dim, num_targets, task_type="regression")
        self.net = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.Linear(input_dim // 2, num_targets),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# Scenario Context Probe
# ---------------------------------------------------------------------------

class ScenarioProbe(BaseProbe):
    """
    Probes whether embeddings preserve scenario/task context.
    Target: scenario identity (multi-class classification).
    """

    def __init__(self, input_dim: int, num_scenarios: int):
        super().__init__(input_dim, num_scenarios, task_type="classification")
        self.net = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.Linear(input_dim // 2, num_scenarios),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# Probe Factory
# ---------------------------------------------------------------------------

def create_probe(probe_type: str, input_dim: int, **kwargs) -> BaseProbe:
    """Create a diagnostic probe by name.

    Args:
        probe_type: 'spatial', 'direction', 'pressure', 'formation',
                    'goal_geometry', 'passing', 'scenario'
        input_dim: dimension of input embeddings
        **kwargs: probe-specific arguments

    Returns:
        BaseProbe instance
    """
    if probe_type == "spatial":
        return SpatialGeometryProbe(input_dim, **kwargs)
    elif probe_type == "direction":
        return DirectionProbe(input_dim, **kwargs)
    elif probe_type == "pressure":
        return PressureProbe(input_dim, **kwargs)
    elif probe_type == "formation":
        return FormationProbe(input_dim, **kwargs)
    elif probe_type == "goal_geometry":
        return GoalGeometryProbe(input_dim, **kwargs)
    elif probe_type == "passing":
        return PassingProbe(input_dim, **kwargs)
    elif probe_type == "scenario":
        return ScenarioProbe(input_dim, **kwargs)
    else:
        raise ValueError(f"Unknown probe type: {probe_type}")
