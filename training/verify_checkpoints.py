#!/usr/bin/env python3
"""
verify_checkpoints.py — SHA-256 and shape audit for GMN-Football-3 checkpoints.
"""

import hashlib
import os
import sys
import torch
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = REPO_ROOT / "training" / "models"


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def inspect_checkpoint(path: Path) -> dict:
    ckpt = torch.load(path, map_location="cpu")
    actor_state = ckpt.get("actor", {})
    first_key = next(iter(actor_state)) if actor_state else None
    return {
        "timesteps": ckpt.get("timesteps"),
        "obs_dim": ckpt.get("obs_dim"),
        "action_dim": ckpt.get("action_dim"),
        "first_param_key": first_key,
        "first_param_shape": list(actor_state[first_key].shape) if first_key else None,
    }


def main():
    print("=" * 80)
    print("CHECKPOINT SHA-256 & SHAPE AUDIT")
    print("=" * 80)

    pt_files = sorted(MODELS_DIR.glob("*.pt"))
    if not pt_files:
        print(f"No .pt files found in {MODELS_DIR}")
        return 1

    rows = []
    hashes = {}
    for pt in pt_files:
        sha = sha256_of(pt)
        hashes[pt.name] = sha
        try:
            meta = inspect_checkpoint(pt)
        except Exception as e:
            meta = {"error": str(e)}
        rows.append((pt.name, sha, meta))

    # Print table
    print(f"\n{'File':<60} {'SHA-256':<20}")
    print("-" * 80)
    for name, sha, _ in rows:
        print(f"{name:<60} {sha[:20]}...")

    # Detect duplicates
    seen = {}
    duplicates = []
    for name, sha, _ in rows:
        if sha in seen:
            duplicates.append((seen[sha], name))
        else:
            seen[sha] = name

    if duplicates:
        print("\n" + "=" * 80)
        print("DUPLICATE CHECKPOINT FILES DETECTED:")
        print("=" * 80)
        for a, b in duplicates:
            print(f"  {a} == {b}")
    else:
        print("\nNo duplicate checkpoint files detected.")

    # Print metadata
    print("\n" + "=" * 80)
    print("CHECKPOINT METADATA")
    print("=" * 80)
    for name, _, meta in rows:
        print(f"\n{name}:")
        for k, v in meta.items():
            print(f"  {k}: {v}")

    print("\n" + "=" * 80)
    print("AUDIT COMPLETE")
    print("=" * 80)
    return 0


if __name__ == "__main__":
    sys.exit(main())
