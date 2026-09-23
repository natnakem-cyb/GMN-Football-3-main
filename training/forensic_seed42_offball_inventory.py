"""READ-ONLY forensic inventory of the canonical reconciled detail artifact.

Purpose
-------
Forensically audit the Seed-42 "off-ball PASS+SHOT" claim directly against the
immutable canonical raw detail artifact:

    training/results/post_reweight_logit_prestep_reconciled_detail.json

It reconstructs, from RAW FRAMES ONLY: per-seed decision / on-ball / PASS+SHOT
counts, the exact off-ball PASS+SHOT frame set (Seed 42: expected 49),
tick / agent / action / mask / ball-owner / team-possession / ep_seed
distributions for that set, temporal-alignment status, canonical-scope
conformance per record, and the occupancy two-component decomposition.

Hard guarantees
---------------
  * read-only: writes nothing to any canonical artifact
  * no training, no environment interaction, no re-collection
  * streams the 326 MB JSON (never calls json.load on the whole document)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from typing import Any, Dict, Iterator, List, Optional, Tuple

DETAIL_PATH = os.path.join(
    "training", "results", "post_reweight_logit_prestep_reconciled_detail.json"
)

PASS_ACTION_IDS = (9, 10, 11)
SHOT_ACTION_IDS = (12,)
PASS_SHOT_ACTION_IDS = (9, 10, 11, 12)
ACTION_DIM = 19
NUM_AGENTS = 3
ACTION_NAMES = [
    "IDLE", "LEFT", "RIGHT", "UP", "DOWN",
    "UP_LEFT", "UP_RIGHT", "DOWN_LEFT", "DOWN_RIGHT",
    "LONG_PASS", "HIGH_PASS", "SHORT_PASS",
    "SHOT", "SPRINT",
    "RELEASE_DIRECTION", "RELEASE_SPRINT",
    "SLIDING", "DRIBBLE", "RELEASE_DRIBBLE",
]

# ---------------------------------------------------------------------------
# Streaming frame iteration (no whole-document json.load)
# ---------------------------------------------------------------------------
def iter_top_level_array_items(text: str, key: str) -> Iterator[str]:
    """Yield the raw JSON text of each element of the top-level array `key`.

    A single linear scan tracks JSON string/escape state and bracket depth, so
    element boundaries are located without building Python objects for the
    whole document.
    """
    n = len(text)
    i = 0
    depth = 0
    in_str = False
    esc = False
    expect_key = False
    while i < n:
        ch = text[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            i += 1
            continue
        if ch == '"':
            if depth == 1 and expect_key:
                j = i + 1
                buf: List[str] = []
                esc2 = False
                while j < n:
                    cj = text[j]
                    if esc2:
                        buf.append(cj)
                        esc2 = False
                    elif cj == "\\":
                        esc2 = True
                    elif cj == '"':
                        break
                    else:
                        buf.append(cj)
                    j += 1
                name = "".join(buf)
                i = j + 1
                expect_key = False
                if name == key:
                    k = i
                    while k < n and text[k] in " \t\r\n":
                        k += 1
                    if k < n and text[k] == ":":
                        k += 1
                        while k < n and text[k] in " \t\r\n":
                            k += 1
                        if k < n and text[k] == "[":
                            yield from _iter_array_elements(text, k)
                            return
                continue
            in_str = True
            i += 1
            continue
        if ch in "{[":
            depth += 1
            if depth == 1:
                expect_key = True
            i += 1
            continue
        if ch in "}]":
            depth -= 1
            i += 1
            continue
        if ch == ",":
            if depth == 1:
                expect_key = True
            i += 1
            continue
        i += 1
    raise KeyError(f"top-level key {key!r} not found")


def _iter_array_elements(text: str, bracket_index: int) -> Iterator[str]:
    """Yield raw JSON text for each element of the array starting at `[`."""
    n = len(text)
    i = bracket_index + 1
    depth = 1  # already inside the array
    start = None
    while i < n:
        ch = text[i]
        if ch == '"':
            i += 1
            while i < n:
                c = text[i]
                if c == "\\":
                    i += 2
                    continue
                if c == '"':
                    i += 1
                    break
                i += 1
            continue
        if ch in "{[":
            if depth == 1:
                start = i
            depth += 1
            i += 1
            continue
        if ch in "}]":
            depth -= 1
            if depth == 0:
                return
            if depth == 1 and start is not None:
                yield text[start : i + 1]
                start = None
            i += 1
            continue
        if depth == 1:
            if ch in " \t\r\n,":
                i += 1
                continue
            s = i
            while i < n and text[i] not in ",]":
                i += 1
            yield text[s:i]
            continue
        i += 1


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _pct(num: int, den: int) -> float:
    return (num / den * 100.0) if den else float("nan")


def _dist(counter: Counter) -> List[Dict[str, Any]]:
    items = sorted(counter.items(), key=lambda kv: (str(type(kv[0])), kv[0]))
    return [{"value": k, "count": v} for k, v in items]


def _mask_legal(mask: List[int]) -> Tuple[List[int], List[int]]:
    legal = [i for i, v in enumerate(mask) if v == 1]
    illegal = [i for i, v in enumerate(mask) if v != 1]
    return legal, illegal


def temporal_alignment(
    frame: Dict[str, Any],
) -> Tuple[bool, Optional[int], Optional[int]]:
    """Canonical temporal-alignment rule used by verify_canonical_artifacts.py.

    expected = argmax_i probs[i] over i where action_mask[i] == 1
    """
    probs = frame.get("probs") or []
    mask = frame.get("action_mask") or []
    action = int(frame["action_taken"])
    legal = [i for i in range(min(ACTION_DIM, len(mask))) if mask[i] == 1]
    if not legal:
        return False, None, action

    def _p(i: int) -> float:
        v = probs[i] if i < len(probs) else None
        if v is None:
            return float("-inf")
        return float(v)

    expected = int(max(legal, key=_p))
    return (action == expected), expected, action



# ---------------------------------------------------------------------------
# Main forensic pass
# ---------------------------------------------------------------------------
def run(detail_path: str, out_path: Optional[str]) -> int:
    print("=" * 78)
    print("SEED-42 OFF-BALL FORENSIC INVENTORY (read-only, raw-frame derived)")
    print("=" * 78)
    print(f"artifact: {detail_path}")
    print(f"bytes   : {os.path.getsize(detail_path)}")
    with open(detail_path, "r", encoding="utf-8") as f:
        text = f.read()
    print(f"chars   : {len(text)}")
    print("head    :", text[:120].replace("\n", " "))
    print()

    per_seed_frames: Counter = Counter()
    per_seed_ps: Counter = Counter()
    per_seed_onball: Counter = Counter()
    per_seed_onball_ps: Counter = Counter()
    per_seed_tick0_frames: Counter = Counter()
    per_seed_tick0_ps: Counter = Counter()
    per_seed_tick0_onball: Counter = Counter()
    per_seed_tick0_onball_ps: Counter = Counter()
    per_seed_offball_ps: Counter = Counter()
    per_seed_offball_ps_by_tick: Dict[int, Counter] = {}
    per_seed_offball_ps_by_agent: Dict[int, Counter] = {}
    per_seed_offball_ps_by_action: Dict[int, Counter] = {}
    per_seed_poss_mask: Counter = Counter()
    per_seed_poss_mask_onball: Counter = Counter()
    per_seed_poss_mask_offball: Counter = Counter()
    per_seed_poss_mask_tick0: Counter = Counter()
    per_seed_poss_mask_tickgt0: Counter = Counter()
    per_seed_ps_with_illegal_ball_actions: Counter = Counter()
    per_seed_tick0_agent0_action: Dict[int, Counter] = {}
    per_seed_tick0_agent0_mask: Dict[int, Counter] = {}
    per_seed_ps_by_agent: Dict[int, Counter] = {}
    per_seed_done_ticks: Dict[int, Counter] = {}
    per_seed_tick0_owner_recorded: Dict[int, Counter] = {}

    seed42_offball_ps: List[Dict[str, Any]] = []
    seed42_onball_ps: List[Dict[str, Any]] = []
    seed42_all_ps: List[Dict[str, Any]] = []
    seed42_tick0_all: List[Dict[str, Any]] = []
    seed42_mask_sum_all: Counter = Counter()
    seed42_mask_sum_nontick0: Counter = Counter()

    n_seen = 0

    for raw in iter_top_level_array_items(text, "frames"):
        fr = json.loads(raw)
        n_seen += 1
        seed = int(fr["seed"])
        per_seed_frames[seed] += 1
        tick = int(fr["tick"])
        action = int(fr["action_taken"])
        is_ps = action in PASS_SHOT_ACTION_IDS
        onball = bool(fr["onball"])
        mask_row = [int(v) for v in fr["action_mask"]]
        # Engine possession predicate (src/engine/ObservationEncoder.ts:395-415):
        # BALL_ACTIONS {9,10,11,12} are legal iff the acting player has possession.
        mask_grants_possession = all(mask_row[i] == 1 for i in (9, 10, 11, 12))
        if mask_grants_possession:
            per_seed_poss_mask[seed] += 1
            if onball:
                per_seed_poss_mask_onball[seed] += 1
            else:
                per_seed_poss_mask_offball[seed] += 1
            if tick == 0:
                per_seed_poss_mask_tick0[seed] += 1
            else:
                per_seed_poss_mask_tickgt0[seed] += 1
        if is_ps and not mask_grants_possession:
            per_seed_ps_with_illegal_ball_actions[seed] += 1
        if tick == 0 and int(fr["agent_index"]) == 0:
            per_seed_tick0_agent0_action.setdefault(seed, Counter())[action] += 1
            per_seed_tick0_agent0_mask.setdefault(seed, Counter())[
                tuple(mask_row)
            ] += 1
        if is_ps:
            per_seed_ps_by_agent.setdefault(seed, Counter())[
                int(fr["agent_index"])
            ] += 1
        if bool(fr["done"]):
            per_seed_done_ticks.setdefault(seed, Counter())[tick] += 1
        if tick == 0:
            per_seed_tick0_owner_recorded.setdefault(seed, Counter())[
                (int(fr["agent_index"]), int(fr["pre_step_ball_owner_agent_idx"]))
            ] += 1
        if seed == 42:
            seed42_mask_sum_all[int(fr["mask_sum"])] += 1
            if tick > 0:
                seed42_mask_sum_nontick0[int(fr["mask_sum"])] += 1
        if is_ps:
            per_seed_ps[seed] += 1
        if onball:
            per_seed_onball[seed] += 1
            if is_ps:
                per_seed_onball_ps[seed] += 1
        if tick == 0:
            per_seed_tick0_frames[seed] += 1
            if is_ps:
                per_seed_tick0_ps[seed] += 1
            if onball:
                per_seed_tick0_onball[seed] += 1
                if is_ps:
                    per_seed_tick0_onball_ps[seed] += 1
        if (not onball) and is_ps:
            per_seed_offball_ps[seed] += 1
            per_seed_offball_ps_by_tick.setdefault(seed, Counter())[tick] += 1
            per_seed_offball_ps_by_agent.setdefault(seed, Counter())[
                int(fr["agent_index"])
            ] += 1
            per_seed_offball_ps_by_action.setdefault(seed, Counter())[action] += 1
        if seed == 42:
            if is_ps:
                seed42_all_ps.append(
                    {
                        "episode": int(fr["episode"]),
                        "ep_seed": int(fr["ep_seed"]),
                        "tick": tick,
                        "agent_index": int(fr["agent_index"]),
                        "agent_id": fr["agent_id"],
                        "action_taken": action,
                        "action_taken_name": fr["action_taken_name"],
                        "onball": onball,
                        "team_has_ball": bool(fr["team_has_ball"]),
                        "pre_step_obs95": float(fr["pre_step_obs95"]),
                        "pre_step_ball_owner_agent_idx": int(
                            fr["pre_step_ball_owner_agent_idx"]
                        ),
                        "mask_sum": int(fr["mask_sum"]),
                    }
                )
                if not onball:
                    rec = dict(fr)
                    aligned, expected, _ = temporal_alignment(fr)
                    rec["_temporal_aligned"] = aligned
                    rec["_expected_argmax"] = expected
                    seed42_offball_ps.append(rec)
                else:
                    seed42_onball_ps.append(dict(fr))
            if tick == 0:
                seed42_tick0_all.append(
                    {
                        "episode": int(fr["episode"]),
                        "agent_index": int(fr["agent_index"]),
                        "agent_id": fr["agent_id"],
                        "action_taken": action,
                        "action_taken_name": fr["action_taken_name"],
                        "onball": onball,
                        "team_has_ball": bool(fr["team_has_ball"]),
                        "pre_step_obs95": float(fr["pre_step_obs95"]),
                        "pre_step_ball_owner_agent_idx": int(
                            fr["pre_step_ball_owner_agent_idx"]
                        ),
                        "mask_sum": int(fr["mask_sum"]),
                        "mask": [int(v) for v in fr["action_mask"]],
                        "team_has_ball_flag": bool(fr["team_has_ball"]),
                        "aligned": temporal_alignment(fr)[0],
                    }
                )

    print(f"total frames streamed           : {n_seen}")
    print(f"seeds present                   : {sorted(per_seed_frames)}")
    print()

    # ---------------- per-seed reconciliation -----------------
    print("-" * 78)
    print("PER-SEED RAW RECONSTRUCTION")
    print("-" * 78)
    print(
        f"{'seed':>5} {'frames':>7} {'onball':>7} {'pass_shot':>9} {'onb_PS':>7} "
        f"{'off_PS':>7} {'tick0_fr':>8} {'tick0_PS':>8} {'t0_onb':>7} {'t0_onbPS':>9}"
    )
    for seed in sorted(per_seed_frames):
        print(
            f"{seed:>5} {per_seed_frames[seed]:>7} {per_seed_onball[seed]:>7} "
            f"{per_seed_ps[seed]:>9} {per_seed_onball_ps[seed]:>7} "
            f"{per_seed_offball_ps[seed]:>7} {per_seed_tick0_frames[seed]:>8} "
            f"{per_seed_tick0_ps[seed]:>8} {per_seed_tick0_onball[seed]:>7} "
            f"{per_seed_tick0_onball_ps[seed]:>9}"
        )
    print()
    for seed in sorted(per_seed_frames):
        n = per_seed_frames[seed]
        print(
            f"seed {seed}: rate = {per_seed_ps[seed]}/{n} = "
            f"{_pct(per_seed_ps[seed], n):.6f}% | "
            f"onball contribution = {per_seed_onball_ps[seed]}/{n} = "
            f"{_pct(per_seed_onball_ps[seed], n):.6f}% | "
            f"offball contribution = {per_seed_offball_ps[seed]}/{n} = "
            f"{_pct(per_seed_offball_ps[seed], n):.6f}%"
        )
    print()

    # ---------------- seed 42 off-ball PS set ------------------
    print("-" * 78)
    print("SEED 42 OFF-BALL PASS+SHOT FRAME SET")
    print("-" * 78)
    print(f"len(seed42_all_ps)        = {len(seed42_all_ps)}")
    print(f"len(seed42_onball_ps)     = {len(seed42_onball_ps)}")
    print(f"len(seed42_offball_ps)    = {len(seed42_offball_ps)}")
    print(
        f"identity check 57 = 8 + 49 : "
        f"{len(seed42_all_ps) == len(seed42_onball_ps) + len(seed42_offball_ps)} "
        f"({len(seed42_onball_ps)} + {len(seed42_offball_ps)} "
        f"= {len(seed42_onball_ps) + len(seed42_offball_ps)})"
    )
    print()

    for label, frameset in (
        ("tick", Counter(int(f["tick"]) for f in seed42_offball_ps)),
        ("agent_index", Counter(int(f["agent_index"]) for f in seed42_offball_ps)),
        ("agent_id", Counter(str(f["agent_id"]) for f in seed42_offball_ps)),
        (
            "action_taken",
            Counter(int(f["action_taken"]) for f in seed42_offball_ps),
        ),
        (
            "action_taken_name",
            Counter(str(f["action_taken_name"]) for f in seed42_offball_ps),
        ),
        ("mask_sum", Counter(int(f["mask_sum"]) for f in seed42_offball_ps)),
        (
            "pre_step_ball_owner_agent_idx",
            Counter(
                int(f["pre_step_ball_owner_agent_idx"]) for f in seed42_offball_ps
            ),
        ),
        (
            "pre_step_obs95",
            Counter(float(f["pre_step_obs95"]) for f in seed42_offball_ps),
        ),
        (
            "team_has_ball",
            Counter(bool(f["team_has_ball"]) for f in seed42_offball_ps),
        ),
        (
            "agent_has_ball",
            Counter(bool(f["agent_has_ball"]) for f in seed42_offball_ps),
        ),
        ("onball", Counter(bool(f["onball"]) for f in seed42_offball_ps)),
        ("episode", Counter(int(f["episode"]) for f in seed42_offball_ps)),
        ("ep_seed", Counter(int(f["ep_seed"]) for f in seed42_offball_ps)),
        ("event_code", Counter(f["event_code"] for f in seed42_offball_ps)),
        ("done", Counter(bool(f["done"]) for f in seed42_offball_ps)),
        ("terminated", Counter(bool(f["terminated"]) for f in seed42_offball_ps)),
        ("truncated", Counter(bool(f["truncated"]) for f in seed42_offball_ps)),
        (
            "deterministic_action_is_pass_shot",
            Counter(
                bool(f["deterministic_action_is_pass_shot"])
                for f in seed42_offball_ps
            ),
        ),
        (
            "temporal_aligned",
            Counter(bool(f["_temporal_aligned"]) for f in seed42_offball_ps),
        ),
        (
            "checkpoint_timesteps",
            Counter(int(f["checkpoint_timesteps"]) for f in seed42_offball_ps),
        ),
        (
            "deterministic_flag",
            Counter(bool(f["deterministic"]) for f in seed42_offball_ps),
        ),
        ("scenario", Counter(str(f["scenario"]) for f in seed42_offball_ps)),
        ("checkpoint", Counter(str(f["checkpoint"]) for f in seed42_offball_ps)),
        (
            "checkpoint_sha256",
            Counter(str(f["checkpoint_sha256"]) for f in seed42_offball_ps),
        ),
        ("code_commit", Counter(str(f["code_commit"]) for f in seed42_offball_ps)),
        (
            "pre_step_obs_ball_ownership_slice",
            Counter(
                tuple(float(v) for v in f["pre_step_obs_ball_ownership_slice"])
                for f in seed42_offball_ps
            ),
        ),
    ):
        print(f"{label} distribution (n={len(seed42_offball_ps)}):")
        for entry in _dist(frameset):
            print(f"    {entry['value']!r}: {entry['count']}")
        print()

    # ---------------- mask vector distribution -----------------
    print("-" * 78)
    print("MASK INVESTIGATION")
    print("-" * 78)
    print("mask VECTOR distribution (seed 42 off-ball PASS+SHOT):")
    vec_counter: Counter = Counter()
    for f in seed42_offball_ps:
        vec_counter[tuple(int(v) for v in f["action_mask"])] += 1
    for vec, cnt in vec_counter.most_common():
        legal, illegal = _mask_legal(list(vec))
        print(f"    vector   : {list(vec)}")
        print(f"    mask_sum : {sum(vec)}")
        print(f"    legal    : {legal}")
        print(
            f"    illegal  : {illegal} "
            f"({[ACTION_NAMES[i] for i in illegal if 0 <= i < len(ACTION_NAMES)]})"
        )
        print(f"    count    : {cnt}")
        print(f"    all_ones : {sum(vec) == ACTION_DIM}")
        print()
    print("mask_sum distribution, ALL seed-42 frames:")
    for ms, cnt in sorted(seed42_mask_sum_all.items()):
        print(f"    mask_sum={ms}: {cnt}")
    print()
    print("mask_sum distribution, seed-42 NON-tick-0 frames:")
    for ms, cnt in sorted(seed42_mask_sum_nontick0.items()):
        print(f"    mask_sum={ms}: {cnt}")
    print()
    print("mask VECTOR distribution, seed-42 NON-tick-0 PASS+SHOT frames:")
    ntv: Counter = Counter()
    for f in seed42_all_ps:
        if int(f["tick"]) > 0:
            ntv[int(f["mask_sum"])] += 1
    for ms, cnt in sorted(ntv.items()):
        print(f"    mask_sum={ms}: {cnt}")
    print()

    # ---------------- tick-0 detail -----------------
    print("-" * 78)
    print("SEED 42 TICK-0 FRAME DETAIL (all 3 agents, all actions)")
    print("-" * 78)
    print(f"tick0 frames total: {len(seed42_tick0_all)}")
    print(
        "tick0 by (agent_index, agent_id): "
        f"{dict(Counter((int(f['agent_index']), str(f['agent_id'])) for f in seed42_tick0_all))}"
    )
    print(
        "tick0 episodes                   : "
        f"{len(set(int(f['episode']) for f in seed42_tick0_all))}"
    )
    print(
        "tick0 owner distribution         : "
        f"{dict(Counter(int(f['pre_step_ball_owner_agent_idx']) for f in seed42_tick0_all))}"
    )
    print(
        "tick0 mask-sum distribution      : "
        f"{dict(Counter(int(f['mask_sum']) for f in seed42_tick0_all))}"
    )
    print(
        "tick0 mask VECTOR distribution   : "
        f"{dict(Counter(tuple(int(v) for v in f['mask']) for f in seed42_tick0_all))}"
    )
    print(
        "tick0 obs95 distribution         : "
        f"{dict(Counter(float(f['pre_step_obs95']) for f in seed42_tick0_all))}"
    )
    print(
        "tick0 team_has_ball distribution : "
        f"{dict(Counter(bool(f['team_has_ball']) for f in seed42_tick0_all))}"
    )
    print(
        "tick0 onball distribution        : "
        f"{dict(Counter(bool(f['onball']) for f in seed42_tick0_all))}"
    )
    print(
        "tick0 temporal_aligned           : "
        f"{dict(Counter(bool(f['aligned']) for f in seed42_tick0_all))}"
    )
    print(
        "tick0 action distribution (agent 0): "
        f"{dict(Counter(int(f['action_taken']) for f in seed42_tick0_all if int(f['agent_index']) == 0))}"
    )
    print(
        "tick0 action distribution (agents 1-2): "
        f"{dict(Counter(int(f['action_taken']) for f in seed42_tick0_all if int(f['agent_index']) != 0))}"
    )
    print()
    print("per-episode tick-0 agent-0 action (seed 42):")
    a0 = sorted(
        (int(f["episode"]), int(f["action_taken"])) for f in seed42_tick0_all if int(f["agent_index"]) == 0
    )
    print("   ", a0)
    print()

    # ---------------- canonical-scope check for the 49 -----------------
    print("-" * 78)
    print("CANONICAL-SCOPE CONFORMANCE OF THE 49 OFF-BALL PASS+SHOT FRAMES")
    print("-" * 78)
    scope_violations: List[str] = []
    for f in seed42_offball_ps:
        tag = (
            f"ep{f['episode']} tick{f['tick']} agent{f['agent_index']} "
            f"action{f['action_taken']}"
        )
        if int(f["seed"]) != 42:
            scope_violations.append(f"{tag}: seed != 42")
        if not (0 <= int(f["episode"]) <= 49):
            scope_violations.append(f"{tag}: episode out of 0..49")
        if not (0 <= int(f["tick"]) <= 50):
            scope_violations.append(f"{tag}: tick out of 0..50")
        if int(f["agent_index"]) not in (0, 1, 2):
            scope_violations.append(f"{tag}: agent_index out of 0..2")
        if int(f["action_taken"]) not in PASS_SHOT_ACTION_IDS:
            scope_violations.append(f"{tag}: action not PASS+SHOT")
        if f["pre_step_obs95"] is None:
            scope_violations.append(f"{tag}: missing pre_step_obs95")
        if not f["action_mask"] or len(f["action_mask"]) != ACTION_DIM:
            scope_violations.append(f"{tag}: action_mask missing/short")
        if not f["deterministic"]:
            scope_violations.append(f"{tag}: deterministic flag false")
        if int(f["checkpoint_timesteps"]) != 49920:
            scope_violations.append(f"{tag}: checkpoint_timesteps != 49920")
        if str(f["scenario"]) != "academy_3_vs_1_with_keeper_onball":
            scope_violations.append(f"{tag}: scenario mismatch")
        if int(f["ep_seed"]) != 500000 + int(f["episode"]) * 1009:
            scope_violations.append(f"{tag}: ep_seed formula mismatch")
        if int(f["action_taken"]) != int(f["_expected_argmax"]):
            scope_violations.append(
                f"{tag}: action != masked argmax ({f['_expected_argmax']})"
            )
    print(f"scope violations found: {len(scope_violations)}")
    for v in scope_violations[:80]:
        print("   ", v)
    print()

    # ---------------- occupancy decomposition -----------------
    print("-" * 78)
    print("SEED 42 OCCUPANCY DECOMPOSITION (raw-frame derived)")
    print("-" * 78)
    N = int(per_seed_frames[42])
    n_on = int(per_seed_onball[42])
    n_off = N - n_on
    on_ps = int(per_seed_onball_ps[42])
    off_ps = int(per_seed_offball_ps[42])
    tot_ps = on_ps + off_ps
    print(f"N (decisions)                 = {N}")
    print(f"n_onball                      = {n_on}  P(onball)={n_on / N:.10f}")
    print(f"n_offball                     = {n_off}  P(offball)={n_off / N:.10f}")
    print(f"on-ball PS selections         = {on_ps}")
    print(f"off-ball PS selections        = {off_ps}")
    print(f"total PS selections           = {tot_ps}")
    print(f"P(PS | onball)                = {on_ps}/{n_on} = {on_ps / max(1, n_on):.10f}")
    print(f"P(PS | offball)               = {off_ps}/{n_off} = {off_ps / max(1, n_off):.10f}")
    print(f"on-ball contribution to rate  = {on_ps}/{N} = {_pct(on_ps, N):.6f}%")
    print(f"off-ball contribution to rate = {off_ps}/{N} = {_pct(off_ps, N):.6f}%")
    print(f"total unconditional rate      = {tot_ps}/{N} = {_pct(tot_ps, N):.6f}%")
    lhs = (n_on / N) * (on_ps / max(1, n_on)) + (n_off / N) * (
        off_ps / max(1, n_off)
    )
    print(
        f"decomposition check           = {lhs:.12f} vs {tot_ps / N:.12f} "
        f"match={abs(lhs - tot_ps / N) < 1e-12}"
    )
    print()
    print("Arithmetic validation (percent):")
    for num in (8, 13, 49, 57):
        print(f"  {num}/7650 = {num / 7650 * 100:.10f}%")
    print()

    # ---------------- all-seed off-ball tick distributions -----------------
    print("-" * 78)
    print("OFF-BALL PASS+SHOT DISTRIBUTIONS, ALL SEEDS")
    print("-" * 78)
    for seed in sorted(per_seed_frames):
        bt = per_seed_offball_ps_by_tick.get(seed, Counter())
        ba = per_seed_offball_ps_by_agent.get(seed, Counter())
        bac = per_seed_offball_ps_by_action.get(seed, Counter())
        print(f"seed {seed}: n_off_PS={per_seed_offball_ps[seed]}")
        print(f"   ticks   : {dict(sorted(bt.items()))}")
        print(f"   agents  : {dict(sorted(ba.items()))}")
        print(f"   actions : {dict(sorted(bac.items()))}")
    print()

    # ---------------- engine-possession cross-check -----------------
    print("-" * 78)
    print("ENGINE-POSSESSION MASK CROSS-CHECK (per seed)")
    print("-" * 78)
    print("A frame whose mask has BALL_ACTIONS {9,10,11,12} all legal is a frame in")
    print("which the ENGINE gave that player possession")
    print("(src/engine/ObservationEncoder.ts:395-415; src/engine/GameEngine.ts:493-501).")
    print()
    print(
        f"{'seed':>5} {'poss_mask':>10} {'recorded_onball':>16} "
        f"{'poss&onball':>12} {'poss&offball(recorded)':>23} {'tick0_poss':>11} "
        f"{'tick>0_poss':>12} {'PS_with_illegal_ball':>21}"
    )
    for seed in sorted(per_seed_frames):
        print(
            f"{seed:>5} {per_seed_poss_mask[seed]:>10} "
            f"{per_seed_onball[seed]:>16} {per_seed_poss_mask_onball[seed]:>12} "
            f"{per_seed_poss_mask_offball[seed]:>23} "
            f"{per_seed_poss_mask_tick0[seed]:>11} "
            f"{per_seed_poss_mask_tickgt0[seed]:>12} "
            f"{per_seed_ps_with_illegal_ball_actions[seed]:>21}"
        )
    print()
    for seed in sorted(per_seed_frames):
        print(
            f"seed {seed}: recorded n_onball={per_seed_onball[seed]}, "
            f"engine-possession-mask frames={per_seed_poss_mask[seed]}, "
            f"difference={per_seed_poss_mask[seed] - per_seed_onball[seed]}, "
            f"tick0 frames among them={per_seed_poss_mask_tick0[seed]}"
        )
    print()
    print("tick-0 agent-0 action distribution per seed:")
    for seed in sorted(per_seed_tick0_agent0_action):
        print(f"   seed {seed}: {dict(sorted(per_seed_tick0_agent0_action[seed].items()))}")
    print()
    print("tick-0 agent-0 mask vector per seed:")
    for seed in sorted(per_seed_tick0_agent0_mask):
        for vec, cnt in per_seed_tick0_agent0_mask[seed].most_common():
            legal, illegal = _mask_legal(list(vec))
            print(f"   seed {seed}: {list(vec)} sum={sum(vec)} illegal={illegal} count={cnt}")
    print()
    # ---------------- additional scope/provenance evidence -----------------
    print("-" * 78)
    print("ADDITIONAL PROVENANCE EVIDENCE")
    print("-" * 78)
    print("PASS+SHOT selections by agent_index, per seed:")
    for seed in sorted(per_seed_ps_by_agent):
        print(f"   seed {seed}: {dict(sorted(per_seed_ps_by_agent[seed].items()))}")
    print()
    print("Tick distribution of frames with done==True (episode terminal frames):")
    for seed in sorted(per_seed_done_ticks):
        print(f"   seed {seed}: {dict(sorted(per_seed_done_ticks[seed].items()))}")
    print()
    print("Tick-0 (agent_index, recorded pre_step_ball_owner_agent_idx) counts:")
    for seed in sorted(per_seed_tick0_owner_recorded):
        pairs = dict(sorted(per_seed_tick0_owner_recorded[seed].items()))
        print(f"   seed {seed}: {pairs}")
    print()


    # ---------------- seed 42 on-ball PS inventory -----------------
    print("-" * 78)
    print(f"SEED 42 ON-BALL PASS+SHOT INVENTORY (n={len(seed42_onball_ps)})")
    print("-" * 78)
    for f in sorted(
        seed42_onball_ps, key=lambda x: (x["episode"], x["tick"], x["agent_index"])
    ):
        print(
            f"   ep={f['episode']:>3} tick={f['tick']:>3} agent={f['agent_index']} "
            f"action={f['action_taken']:>2} ({f['action_taken_name']:<12}) "
            f"owner={f['pre_step_ball_owner_agent_idx']} obs95={f['pre_step_obs95']} "
            f"mask_sum={f['mask_sum']} team_has_ball={f['team_has_ball']}"
        )
    print()

    # ---------------- full 49-frame inventory dump -----------------
    if out_path:
        payload = {
            "artifact": detail_path,
            "n_frames_streamed": n_seen,
            "per_seed": {
                str(s): {
                    "n_decisions": int(per_seed_frames[s]),
                    "n_onball": int(per_seed_onball[s]),
                    "n_pass_shot": int(per_seed_ps[s]),
                    "n_onball_pass_shot": int(per_seed_onball_ps[s]),
                    "n_offball_pass_shot": int(per_seed_offball_ps[s]),
                    "n_tick0_frames": int(per_seed_tick0_frames[s]),
                    "n_tick0_pass_shot": int(per_seed_tick0_ps[s]),
                    "canonical_rate_pct": _pct(per_seed_ps[s], per_seed_frames[s]),
                }
                for s in sorted(per_seed_frames)
            },
            "seed42_offball_pass_shot_frames": seed42_offball_ps,
        }
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"wrote forensic inventory: {out_path}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--detail", default=DETAIL_PATH)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    return run(args.detail, args.out)


if __name__ == "__main__":
    sys.exit(main())

