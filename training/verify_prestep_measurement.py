"""
GMN-Football-3 — Independent consistency check for the PRE-STEP on-ball artifacts.

Reads ``training/results/post_reweight_logit_prestep_detail.json`` and
recomputes every aggregate reported in the summary/occupancy CSVs straight from
the raw per-frame records. The pi_* recomputation deliberately uses its own
softmax implementation instead of importing ``training.prestep_onball``, so this
is an independent path rather than a re-run of the collector's own helper.

Checks performed
----------------
1. detail JSON parses and contains no literal Infinity / NaN.
2. checkpoint SHA-256 fields are non-empty and match the committed inventory.
3. row counts match the expected number of checkpoints (4).
4. every retained frame has obs95 == 1.0 (pre-step retention invariant).
5. n_ticks reconciles between the per-episode records and the aggregate row.
6. n_onball / P(on-ball) recompute from raw frames.
7. legality counts/rates recompute from the raw ``action_mask`` vector.
8. pi_PASS / pi_SHOT / pi_PASS+SHOT recompute from raw logits + mask.
9. deterministic action == argmax of the recomputed probabilities.
10. behavioural PASS+SHOT selection count recomputes and stays distinct from pi.
11. summary + occupancy CSVs equal the recomputed values (within float tolerance).

Exit code 0 = all checks pass; non-zero = at least one check failed.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
from typing import Any, Dict, List, Optional, Sequence

PASS_ACTION_IDS = (9, 10, 11)
SHOT_ACTION_IDS = (12,)
PASS_SHOT_ACTION_IDS = (9, 10, 11, 12)
# Stored probabilities come from a torch float32 forward pass; the independent
# recomputation below runs in Python float64. The tolerance covers that
# precision difference only -- it is far tighter than any substantive drift.
FLOAT_TOL = 1e-6
# The JSON value-position scan for invalid literals (": NaN", ": -Infinity", ...).
INVALID_LITERAL_RE = re.compile(r"[:,\[]\s*-?(?:Infinity|NaN)\b")

FAILURES: List[str] = []


def _reject_constant(name: str) -> Any:
    raise ValueError(f"Invalid JSON constant in artifact: {name}")


def check(label: str, condition: bool, detail: str = "") -> None:
    if condition:
        print(f"  PASS  {label}")
    else:
        message = f"{label}{(' :: ' + detail) if detail else ''}"
        FAILURES.append(message)
        print(f"  FAIL  {message}")


def _softmax(values: Sequence[float]) -> List[float]:
    m = max(values)
    exp = [math.exp(v - m) for v in values]
    total = sum(exp)
    return [e / total for e in exp]


def _recompute_probs(
    raw_logits: Sequence[float], action_mask: Sequence[int]
) -> List[float]:
    """Independent masked softmax: illegal slots are excluded from the support."""
    masked = [
        (float(value) if int(mask) == 1 else None)
        for value, mask in zip(raw_logits, action_mask)
    ]
    legal_values = [v for v in masked if v is not None]
    if not legal_values:
        raise ValueError("action_mask has no legal action")
    probs = _softmax(legal_values)
    out: List[float] = []
    it = iter(probs)
    for value in masked:
        out.append(0.0 if value is None else next(it))
    return out


def _group_sum(probs: Sequence[float], ids: Sequence[int]) -> float:
    return float(sum(float(probs[i]) for i in ids))


def _mean(values: Sequence[float]) -> Optional[float]:
    if not values:
        return None
    return float(sum(values) / len(values))


def _rate(numerator: int, denominator: int) -> Optional[float]:
    if denominator <= 0:
        return None
    return float(numerator) / float(denominator)


def _close(a: Optional[float], b: Optional[float], tol: float = FLOAT_TOL) -> bool:
    if a is None and b is None:
        return True
    if a is None or b is None:
        return False
    return abs(float(a) - float(b)) <= tol


def _read_csv_rows(path: str) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _float_field(row: Dict[str, str], key: str) -> Optional[float]:
    raw = row.get(key, "")
    if raw is None or str(raw).strip() == "":
        return None
    return float(raw)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Verify PRE-STEP on-ball artifacts")
    parser.add_argument(
        "--detail",
        default=os.path.join(
            "training", "results", "post_reweight_logit_prestep_detail.json"
        ),
    )
    parser.add_argument(
        "--summary",
        default=os.path.join(
            "training", "results", "post_reweight_logit_prestep_summary.csv"
        ),
    )
    parser.add_argument(
        "--occupancy",
        default=os.path.join(
            "training", "results", "onball_occupancy_prestep_summary.csv"
        ),
    )
    parser.add_argument(
        "--inventory",
        default=os.path.join("training", "results", "retest_checkpoint_inventory.csv"),
    )
    parser.add_argument("--expected-checkpoints", type=int, default=4)
    args = parser.parse_args(argv)

    print("=" * 72)
    print("INDEPENDENT CONSISTENCY CHECK - PRE-STEP ON-BALL MEASUREMENT")
    print("=" * 72)

    # ---- 1. JSON parses; no invalid literals ------------------------------
    with open(args.detail, "r", encoding="utf-8") as f:
        text = f.read()
    literal_match = INVALID_LITERAL_RE.search(text)
    check(
        "detail JSON contains no Infinity/NaN value literals",
        literal_match is None,
        "" if literal_match is None else f"found {literal_match.group(0)!r}",
    )
    try:
        detail = json.loads(text, parse_constant=_reject_constant)
        parsed_ok = True
    except ValueError as exc:
        detail = {}
        parsed_ok = False
        print(f"  (json.loads rejected the artifact: {exc})")
    check(
        "detail JSON parsed strictly (parse_constant rejects Infinity/NaN)",
        parsed_ok
        and isinstance(detail, dict)
        and "frames" in detail
        and "results" in detail,
    )

    provenance = detail.get("provenance", {})
    results = detail.get("results", [])
    frames = detail.get("frames", [])
    rows = detail.get("aggregate", {}).get("rows", [])

    # ---- 2. SHA provenance -----------------------------------------------
    with open(args.inventory, "r", encoding="utf-8", newline="") as f:
        inventory = {
            int(r["seed"]): r["sha256"].strip()
            for r in csv.DictReader(f)
            if str(r.get("trajectory_step", "")).strip().lower() == "final"
        }
    sha_ok = True
    for ckpt in provenance.get("checkpoints", []):
        sha = ckpt.get("checkpoint_sha256") or ""
        if not sha:
            sha_ok = False
        if inventory.get(int(ckpt["seed"])) != sha:
            sha_ok = False
        if not ckpt.get("sha_verified_against_inventory"):
            sha_ok = False
    check("all checkpoint SHA-256 values non-empty and inventory-verified", sha_ok)

    # ---- 3. Row counts ----------------------------------------------------
    check(
        f"row count == expected checkpoints ({args.expected_checkpoints})",
        len(rows) == args.expected_checkpoints,
        f"rows={len(rows)}",
    )
    check(
        "aggregate rows == results count",
        len(rows) == len(results),
        f"rows={len(rows)} results={len(results)}",
    )

    # ---- 4. Per-seed recomputation from raw frames ------------------------
    frames_by_seed: Dict[int, List[Dict[str, Any]]] = {}
    for fr in frames:
        frames_by_seed.setdefault(int(fr["seed"]), []).append(fr)

    check(
        "every frame onball_source == 'obs[95]_pre_step'",
        all(fr.get("onball_source") == "obs[95]_pre_step" for fr in frames),
    )
    check(
        "every retained frame has obs95 == 1.0",
        all(float(fr["obs95"]) == 1.0 for fr in frames),
    )
    check(
        "every retained frame's ownership slice is [0, 1, 0]",
        all(
            [float(v) for v in fr["obs_ball_ownership_slice"]] == [0.0, 1.0, 0.0]
            for fr in frames
        ),
    )

    summary_rows = {int(r["seed"]): r for r in _read_csv_rows(args.summary)}
    occupancy_rows = {int(r["seed"]): r for r in _read_csv_rows(args.occupancy)}
    check(
        "summary CSV row count == detail row count",
        len(summary_rows) == len(rows),
        f"csv={len(summary_rows)} detail={len(rows)}",
    )
    check(
        "occupancy CSV row count == detail row count",
        len(occupancy_rows) == len(rows),
        f"csv={len(occupancy_rows)} detail={len(rows)}",
    )

    for result, row in zip(results, rows):
        seed = int(result["seed"])
        seed_frames = frames_by_seed.get(seed, [])
        print(f"\n[seed {seed}] raw frames = {len(seed_frames)}")

        # ---- 5/6. ticks and occupancy ------------------------------------
        ticks_from_episodes = sum(int(e["n_ticks"]) for e in result["episodes"])
        check(
            "n_ticks reconciles (episodes vs aggregate row)",
            ticks_from_episodes == int(row["n_ticks"]),
            f"episodes={ticks_from_episodes} row={row['n_ticks']}",
        )
        check(
            "n_onball reconciles (raw frames vs aggregate row)",
            len(seed_frames) == int(row["n_onball"]),
            f"frames={len(seed_frames)} row={row['n_onball']}",
        )
        check(
            "n_onball == sum(episode n_onball_prestep)",
            len(seed_frames)
            == sum(int(e["n_onball_prestep"]) for e in result["episodes"]),
        )
        check(
            "P(on-ball) == n_onball / n_ticks",
            _close(row["p_onball"], _rate(len(seed_frames), int(row["n_ticks"]))),
        )

        # ---- 7. legality from the raw mask -------------------------------
        pass_legal = [
            fr
            for fr in seed_frames
            if any(int(fr["action_mask"][i]) == 1 for i in PASS_ACTION_IDS)
        ]
        shot_legal = [fr for fr in seed_frames if int(fr["action_mask"][12]) == 1]
        either_legal = [
            fr
            for fr in seed_frames
            if int(fr["action_mask"][12]) == 1
            or any(int(fr["action_mask"][i]) == 1 for i in PASS_ACTION_IDS)
        ]
        check(
            "n_pass_legal matches raw mask",
            len(pass_legal) == int(row["n_pass_legal"]),
            f"raw={len(pass_legal)} row={row['n_pass_legal']}",
        )
        check(
            "n_shot_legal matches raw mask",
            len(shot_legal) == int(row["n_shot_legal"]),
            f"raw={len(shot_legal)} row={row['n_shot_legal']}",
        )
        check(
            "n_pass_or_shot_legal matches raw mask",
            len(either_legal) == int(row["n_pass_or_shot_legal"]),
        )
        check(
            "legality rates match recomputed rates",
            _close(row["pass_legal_rate"], _rate(len(pass_legal), len(seed_frames)))
            and _close(
                row["shot_legal_rate"], _rate(len(shot_legal), len(seed_frames))
            )
            and _close(
                row["pass_or_shot_legal_rate"],
                _rate(len(either_legal), len(seed_frames)),
            ),
        )

        # ---- 8/9. pi recomputation from raw logits + mask ----------------
        pi_pass_all: List[float] = []
        pi_shot_all: List[float] = []
        pi_pass_shot_all: List[float] = []
        pi_pass_legal: List[float] = []
        pi_shot_legal: List[float] = []
        pi_pass_shot_either: List[float] = []
        max_prob_err = 0.0
        action_mismatches = 0
        normalisation_errors = 0

        for fr in seed_frames:
            probs = _recompute_probs(fr["raw_logits"], fr["action_mask"])
            if abs(sum(probs) - 1.0) > 1e-6:
                normalisation_errors += 1

            pi_pass = _group_sum(probs, PASS_ACTION_IDS)
            pi_shot = _group_sum(probs, SHOT_ACTION_IDS)
            pi_pass_shot = _group_sum(probs, PASS_SHOT_ACTION_IDS)

            max_prob_err = max(
                max_prob_err,
                abs(pi_pass - float(fr["pi_pass"])),
                abs(pi_shot - float(fr["pi_shot"])),
                abs(pi_pass_shot - float(fr["pi_pass_shot"])),
            )
            argmax_index = max(range(len(probs)), key=lambda i: probs[i])
            if int(fr["action_taken"]) != int(argmax_index):
                action_mismatches += 1

            pi_pass_all.append(pi_pass)
            pi_shot_all.append(pi_shot)
            pi_pass_shot_all.append(pi_pass_shot)
            is_pass_legal = any(int(fr["action_mask"][i]) == 1 for i in PASS_ACTION_IDS)
            is_shot_legal = int(fr["action_mask"][12]) == 1
            if is_pass_legal:
                pi_pass_legal.append(pi_pass)
            if is_shot_legal:
                pi_shot_legal.append(pi_shot)
            if is_pass_legal or is_shot_legal:
                pi_pass_shot_either.append(pi_pass_shot)

        check(
            "recomputed probabilities are normalised",
            normalisation_errors == 0,
            f"bad_rows={normalisation_errors}",
        )
        check(
            "stored pi_* equal independently recomputed pi_*",
            max_prob_err <= 1e-6,
            f"max_abs_err={max_prob_err:.3e}",
        )
        check(
            "deterministic action == argmax of recomputed probabilities",
            action_mismatches == 0,
            f"mismatches={action_mismatches}",
        )
        for column, recomputed in (
            ("mean_pi_pass_all_onball", _mean(pi_pass_all)),
            ("mean_pi_shot_all_onball", _mean(pi_shot_all)),
            ("mean_pi_pass_shot_all_onball", _mean(pi_pass_shot_all)),
            ("mean_pi_pass_when_pass_legal", _mean(pi_pass_legal)),
            ("mean_pi_shot_when_shot_legal", _mean(pi_shot_legal)),
            ("mean_pi_pass_shot_when_either_legal", _mean(pi_pass_shot_either)),
        ):
            check(f"{column} matches recomputation", _close(row[column], recomputed))

        # ---- 10. behavioural statistic (separate from pi) ----------------
        selected_pass_shot = sum(
            1 for fr in seed_frames if int(fr["action_taken"]) in PASS_SHOT_ACTION_IDS
        )
        check(
            "n_pass_shot_actions matches raw selected actions",
            selected_pass_shot == int(row["n_pass_shot_actions"]),
            f"raw={selected_pass_shot} row={row['n_pass_shot_actions']}",
        )
        check(
            "P(selected PASS+SHOT | on-ball) matches recomputation",
            _close(
                row["p_selected_pass_shot_given_onball"],
                _rate(selected_pass_shot, len(seed_frames)),
            ),
        )

        # ---- 11. CSV agreement -------------------------------------------
        csv_row = summary_rows.get(seed)
        check(f"seed {seed} present in summary CSV", csv_row is not None)
        if csv_row is not None:
            check(
                "summary CSV sha256 non-empty and equal to provenance",
                bool(csv_row["checkpoint_sha256"])
                and csv_row["checkpoint_sha256"] == row["checkpoint_sha256"],
            )
            check(
                "summary CSV n_onball/n_ticks match detail row",
                csv_row["n_onball"] == str(row["n_onball"])
                and csv_row["n_ticks"] == str(row["n_ticks"]),
            )
            check(
                "summary CSV pi columns match detail row",
                all(
                    _close(_float_field(csv_row, c), row[c])
                    for c in (
                        "mean_pi_pass_all_onball",
                        "mean_pi_shot_all_onball",
                        "mean_pi_pass_shot_all_onball",
                        "mean_pi_pass_when_pass_legal",
                        "mean_pi_shot_when_shot_legal",
                        "mean_pi_pass_shot_when_either_legal",
                        "p_selected_pass_shot_given_onball",
                    )
                ),
            )
        occ_row = occupancy_rows.get(seed)
        check(f"seed {seed} present in occupancy CSV", occ_row is not None)
        if occ_row is not None:
            check(
                "occupancy CSV n_ticks/n_onball/P(on-ball) match detail row",
                occ_row["n_ticks"] == str(row["n_ticks"])
                and occ_row["n_onball"] == str(row["n_onball"])
                and _close(_float_field(occ_row, "p_onball"), row["p_onball"]),
            )

    print("\n" + "=" * 72)
    if FAILURES:
        print(f"CONSISTENCY CHECK FAILED - {len(FAILURES)} problem(s):")
        for failure in FAILURES:
            print(f"  - {failure}")
        print("=" * 72)
        return 1
    print("CONSISTENCY CHECK PASSED - artifacts are internally consistent")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())