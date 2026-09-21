"""Independent verification for canonical 3-agent reconciled artifacts.

Authoritative verifier for the canonical close-out (commit 2d4b6dd and later).
Validates the reconciled artifacts against the standardized tolerance of ±0.05 pp.

This verifier enforces semantic invariants on the raw detail JSON, not merely
schema presence. It independently reconstructs the summary and reconciliation
CSV values from the raw frames and compares them against the committed artifacts.
"""
import csv
import json
import math
from collections import defaultdict
from typing import Any, Dict, List, Tuple

RECONCILIATION_TOLERANCE_PP = 0.05

PASS_ACTION_IDS = (9, 10, 11)
SHOT_ACTION_IDS = (12,)
PASS_SHOT_ACTION_IDS = (9, 10, 11, 12)
ACTION_DIM = 19
NUM_AGENTS = 3
EXPECTED_SEEDS = [42, 123, 7, 999]
EXPECTED_FRAMES = 30600
EXPECTED_DECISIONS_PER_SEED = 7650
EXPECTED_EPISODES = 50
EXPECTED_TICKS_PER_EPISODE = 51


def _check(description: str, condition: bool, detail: str = "") -> None:
    if not condition:
        raise AssertionError(f"FAIL: {description}" + (f" — {detail}" if detail else ""))
    print(f"PASS: {description}")


def _load_detail(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    detail = json.loads(text)
    return detail


def _load_csv_rows(path: str) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _reconstruct_summary_from_frames(frames: List[Dict[str, Any]]) -> Dict[int, Dict[str, Any]]:
    """Independently reconstruct per-seed summary aggregates from raw frames."""
    by_seed: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for fr in frames:
        by_seed[int(fr["seed"])].append(fr)

    reconstructed = {}
    for seed, seed_frames in by_seed.items():
        n_decisions = len(seed_frames)
        n_ticks_agents = n_decisions // NUM_AGENTS
        n_pass = sum(1 for fr in seed_frames if fr["action_taken"] in PASS_ACTION_IDS)
        n_shot = sum(1 for fr in seed_frames if fr["action_taken"] in SHOT_ACTION_IDS)
        n_pass_shot = n_pass + n_shot
        canonical_rate_pct = (n_pass_shot / max(1, n_decisions)) * 100.0

        onball_frames = [fr for fr in seed_frames if fr["onball"]]
        n_onball = len(onball_frames)
        p_onball = n_onball / max(1, n_decisions)

        agent_occupancy = {}
        for agent_idx in range(NUM_AGENTS):
            agent_frames = [fr for fr in seed_frames if fr["agent_index"] == agent_idx]
            n_agent_decisions = len(agent_frames)
            n_agent_onball = sum(1 for fr in agent_frames if fr["onball"])
            n_agent_selected_ps = sum(
                1 for fr in agent_frames
                if fr["onball"] and fr["action_taken"] in PASS_SHOT_ACTION_IDS
            )
            agent_occupancy[agent_idx] = {
                "n_decisions": n_agent_decisions,
                "n_onball": n_agent_onball,
                "p_onball": n_agent_onball / max(1, n_agent_decisions),
                "n_selected_ps_when_onball": n_agent_selected_ps,
                "p_selected_ps_given_onball": n_agent_selected_ps / max(1, n_agent_onball),
            }

        total_agent_onball = sum(a["n_onball"] for a in agent_occupancy.values())
        total_selected_ps = sum(a["n_selected_ps_when_onball"] for a in agent_occupancy.values())

        reconstructed[seed] = {
            "seed": seed,
            "n_ticks": n_ticks_agents,
            "n_decisions": n_decisions,
            "n_pass": n_pass,
            "n_shot": n_shot,
            "n_pass_shot": n_pass_shot,
            "canonical_rate_pct": canonical_rate_pct,
            "n_onball": n_onball,
            "p_onball": p_onball,
            "agent0_n_decisions": agent_occupancy[0]["n_decisions"],
            "agent0_n_onball": agent_occupancy[0]["n_onball"],
            "agent0_p_onball": agent_occupancy[0]["p_onball"],
            "agent0_n_selected_ps_when_onball": agent_occupancy[0]["n_selected_ps_when_onball"],
            "agent0_p_selected_ps_given_onball": agent_occupancy[0]["p_selected_ps_given_onball"],
            "agent1_n_decisions": agent_occupancy[1]["n_decisions"],
            "agent1_n_onball": agent_occupancy[1]["n_onball"],
            "agent1_p_onball": agent_occupancy[1]["p_onball"],
            "agent1_n_selected_ps_when_onball": agent_occupancy[1]["n_selected_ps_when_onball"],
            "agent1_p_selected_ps_given_onball": agent_occupancy[1]["p_selected_ps_given_onball"],
            "agent2_n_decisions": agent_occupancy[2]["n_decisions"],
            "agent2_n_onball": agent_occupancy[2]["n_onball"],
            "agent2_p_onball": agent_occupancy[2]["p_onball"],
            "agent2_n_selected_ps_when_onball": agent_occupancy[2]["n_selected_ps_when_onball"],
            "agent2_p_selected_ps_given_onball": agent_occupancy[2]["p_selected_ps_given_onball"],
            "team_n_agent_onball_total": total_agent_onball,
            "team_p_agent_onball": total_agent_onball / max(1, n_decisions),
            "team_n_selected_ps_onball_total": total_selected_ps,
            "team_p_selected_ps_given_agent_onball": total_selected_ps / max(1, total_agent_onball),
        }
    return reconstructed


def _reconstruct_reconciliation_from_frames(
    frames: List[Dict[str, Any]],
    rebuilt_rates: Dict[int, float],
) -> List[Dict[str, Any]]:
    """Independently reconstruct reconciliation CSV from raw frames."""
    by_seed: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for fr in frames:
        by_seed[int(fr["seed"])].append(fr)

    recon = []
    for seed in sorted(by_seed.keys()):
        seed_frames = by_seed[seed]
        n_decisions = len(seed_frames)
        n_pass_shot = sum(
            1 for fr in seed_frames if fr["action_taken"] in PASS_SHOT_ACTION_IDS
        )
        canonical_rate_pct = (n_pass_shot / max(1, n_decisions)) * 100.0

        total_agent_onball = sum(
            1 for fr in seed_frames if fr["onball"]
        )
        total_selected_ps = sum(
            1 for fr in seed_frames
            if fr["onball"] and fr["action_taken"] in PASS_SHOT_ACTION_IDS
        )
        onball_conditional_ps_rate = total_selected_ps / max(1, total_agent_onball)

        rebuilt_rate = rebuilt_rates.get(seed)
        delta = None
        match = False
        if rebuilt_rate is not None:
            delta = canonical_rate_pct - rebuilt_rate
            match = abs(delta) <= RECONCILIATION_TOLERANCE_PP

        recon.append({
            "seed": seed,
            "n_decisions": n_decisions,
            "team_wide_n_ps_selected": total_selected_ps,
            "team_wide_canonical_rate_pct": canonical_rate_pct,
            "total_controlled_agent_onball_frames": total_agent_onball,
            "agent_level_onball_ps_selections": total_selected_ps,
            "onball_conditional_ps_rate": onball_conditional_ps_rate,
            "team_wide_rate_from_raw_decisions": canonical_rate_pct,
            "canonical_rebuilt_csv_rate": rebuilt_rate,
            "delta": delta,
            "match": match,
        })
    return recon


def _validate_semantic_invariants(frames: List[Dict[str, Any]]) -> None:
    """Validate on-ball, team-possession, and ownership invariants row-by-row."""
    failures = []
    for fr in frames:
        seed = fr.get("seed", "?")
        episode = fr.get("episode", "?")
        tick = fr.get("tick", "?")
        agent_index = fr.get("agent_index", "?")
        identifier = f"seed={seed} ep={episode} tick={tick} agent={agent_index}"

        pre_step_obs95 = fr.get("pre_step_obs95")
        pre_step_ball_owner_agent_idx = fr.get("pre_step_ball_owner_agent_idx")
        team_has_ball = fr.get("team_has_ball")
        agent_has_ball = fr.get("agent_has_ball")
        onball = fr.get("onball")

        # Invariant 1: team_has_ball == (pre_step_obs95 == 1.0)
        expected_team_has_ball = bool(float(pre_step_obs95) == 1.0)
        if team_has_ball != expected_team_has_ball:
            failures.append(
                f"{identifier}: team_has_ball={team_has_ball} != (pre_step_obs95==1.0)={expected_team_has_ball}"
            )

        # Invariant 2: agent_has_ball == (pre_step_ball_owner_agent_idx == agent_index)
        expected_agent_has_ball = bool(pre_step_ball_owner_agent_idx == agent_index)
        if agent_has_ball != expected_agent_has_ball:
            failures.append(
                f"{identifier}: agent_has_ball={agent_has_ball} != (ball_owner_idx==agent_index)={expected_agent_has_ball}"
            )

        # Invariant 3: onball == agent_has_ball
        if onball != agent_has_ball:
            failures.append(
                f"{identifier}: onball={onball} != agent_has_ball={agent_has_ball}"
            )

        # Invariant 4: onball == (pre_step_ball_owner_agent_idx == agent_index)
        expected_onball = bool(pre_step_ball_owner_agent_idx == agent_index)
        if onball != expected_onball:
            failures.append(
                f"{identifier}: onball={onball} != (ball_owner_idx==agent_index)={expected_onball}"
            )

        if failures:
            print(f"FAIL: semantic invariant violation at {failures[0]}")
            if len(failures) > 1:
                for f in failures[1:]:
                    print(f"      also: {f}")
            raise AssertionError(f"{len(failures)} semantic invariant(s) violated")


def _compare_summary_csv(
    reconstructed: Dict[int, Dict[str, Any]],
    committed_rows: List[Dict[str, str]],
    tolerance: float = 1e-9,
) -> None:
    """Compare reconstructed summary values against committed CSV."""
    committed_by_seed = {int(r["seed"]): r for r in committed_rows}
    assert set(reconstructed.keys()) == set(committed_by_seed.keys()), \
        f"seed mismatch: reconstructed={sorted(reconstructed.keys())} committed={sorted(committed_by_seed.keys())}"

    float_fields = [
        "p_onball",
        "agent0_p_onball",
        "agent0_p_selected_ps_given_onball",
        "agent1_p_onball",
        "agent1_p_selected_ps_given_onball",
        "agent2_p_onball",
        "agent2_p_selected_ps_given_onball",
        "team_p_agent_onball",
        "team_p_selected_ps_given_agent_onball",
        "canonical_rate_pct",
    ]

    for seed, rec in reconstructed.items():
        comm = committed_by_seed[seed]
        for field in ["n_ticks", "n_decisions", "n_pass", "n_shot", "n_pass_shot",
                       "n_onball", "agent0_n_decisions", "agent0_n_onball",
                       "agent0_n_selected_ps_when_onball", "agent1_n_decisions",
                       "agent1_n_onball", "agent1_n_selected_ps_when_onball",
                       "agent2_n_decisions", "agent2_n_onball",
                       "agent2_n_selected_ps_when_onball",
                       "team_n_agent_onball_total", "team_n_selected_ps_onball_total"]:
            rec_val = int(rec[field])
            comm_val = int(comm[field])
            _check(
                f"seed {seed}: {field} matches committed CSV",
                rec_val == comm_val,
                f"reconstructed={rec_val} committed={comm_val}",
            )
        for field in float_fields:
            if field in rec and rec[field] is not None:
                rec_val = float(rec[field])
                comm_val = float(comm.get(field, "")) if comm.get(field, "") != "" else None
                if comm_val is not None:
                    _check(
                        f"seed {seed}: {field} matches committed CSV",
                        abs(rec_val - comm_val) <= tolerance,
                        f"reconstructed={rec_val:.6f} committed={comm_val:.6f}",
                    )


def _compare_reconciliation_csv(
    reconstructed: List[Dict[str, Any]],
    committed_rows: List[Dict[str, str]],
    tolerance: float = RECONCILIATION_TOLERANCE_PP,
) -> None:
    """Compare reconstructed reconciliation values against committed CSV."""
    committed_by_seed = {int(r["seed"]): r for r in committed_rows}
    assert len(reconstructed) == len(committed_rows), \
        f"reconciliation row count mismatch: {len(reconstructed)} vs {len(committed_rows)}"

    for rec in reconstructed:
        seed = rec["seed"]
        comm = committed_by_seed[seed]
        _check(
            f"seed {seed}: n_decisions matches committed reconciliation",
            rec["n_decisions"] == int(comm["n_decisions"]),
            f"reconstructed={rec['n_decisions']} committed={comm['n_decisions']}",
        )
        _check(
            f"seed {seed}: team_wide_n_ps_selected matches committed reconciliation",
            rec["team_wide_n_ps_selected"] == int(comm["team_wide_n_ps_selected"]),
            f"reconstructed={rec['team_wide_n_ps_selected']} committed={comm['team_wide_n_ps_selected']}",
        )
        _check(
            f"seed {seed}: total_controlled_agent_onball_frames matches committed reconciliation",
            rec["total_controlled_agent_onball_frames"] == int(comm["total_controlled_agent_onball_frames"]),
            f"reconstructed={rec['total_controlled_agent_onball_frames']} committed={comm['total_controlled_agent_onball_frames']}",
        )
        _check(
            f"seed {seed}: canonical_rate_pct matches committed reconciliation",
            abs(rec["team_wide_canonical_rate_pct"] - float(comm["team_wide_canonical_rate_pct"])) <= 1e-9,
            f"reconstructed={rec['team_wide_canonical_rate_pct']:.6f} committed={float(comm['team_wide_canonical_rate_pct']):.6f}",
        )
        _check(
            f"seed {seed}: onball_conditional_ps_rate matches committed reconciliation",
            abs(rec["onball_conditional_ps_rate"] - float(comm["onball_conditional_ps_rate"])) <= 1e-9,
            f"reconstructed={rec['onball_conditional_ps_rate']:.6f} committed={float(comm['onball_conditional_ps_rate']):.6f}",
        )

        rebuilt_rate = rec.get("canonical_rebuilt_csv_rate")
        if rebuilt_rate is not None:
            comm_rebuilt = comm.get("canonical_rebuilt_csv_rate", "")
            comm_rebuilt_val = float(comm_rebuilt) if comm_rebuilt != "" else None
            if comm_rebuilt_val is not None:
                _check(
                    f"seed {seed}: rebuilt rate matches committed reconciliation",
                    abs(rebuilt_rate - comm_rebuilt_val) <= 1e-9,
                    f"reconstructed={rebuilt_rate:.3f} committed={comm_rebuilt_val:.3f}",
                )

        delta = rec.get("delta")
        if delta is not None:
            comm_delta = comm.get("delta", "")
            comm_delta_val = float(comm_delta) if comm_delta != "" else None
            if comm_delta_val is not None:
                _check(
                    f"seed {seed}: delta matches committed reconciliation",
                    abs(delta - comm_delta_val) <= 1e-9,
                    f"reconstructed={delta:.6f} committed={comm_delta_val:.6f}",
                )

        match = rec.get("match")
        comm_match = comm.get("match", "")
        if match is not None and comm_match != "":
            _check(
                f"seed {seed}: match flag matches committed reconciliation",
                bool(match) == (comm_match in ("1", "True", "true", "yes")),
                f"reconstructed={match} committed={comm_match}",
            )


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser(description="Canonical artifact verifier")
    parser.add_argument("--detail", default="training/results/post_reweight_logit_prestep_reconciled_detail.json")
    parser.add_argument("--summary", default="training/results/post_reweight_logit_prestep_reconciled_summary.csv")
    parser.add_argument("--recon", default="training/results/post_reweight_canonical_scope_reconciliation.csv")
    args = parser.parse_args()

    detail_path = args.detail
    summary_path = args.summary
    recon_path = args.recon

    # Guard against accidentally validating 085ec85-era artifacts
    historical_markers = [
        "post_reweight_logit_prestep_detail.json",
        "post_reweight_logit_prestep_summary.csv",
        "onball_occupancy_summary.csv",
        "onball_occupancy_prestep_summary.csv",
    ]
    for marker in historical_markers:
        if marker in detail_path or marker in summary_path or marker in recon_path:
            print(f"ERROR: {marker} is a historical 085ec85-era artifact.")
            print("Use verify_canonical_artifacts.py with the reconciled artifact names:")
            print(f"  detail : {detail_path}")
            print(f"  summary: {summary_path}")
            print(f"  recon  : {recon_path}")
            return 1

    print("=== INDEPENDENT VERIFICATION: CANONICAL 3-AGENT ARTIFACTS ===")

    # 1. Load raw detail JSON
    detail = _load_detail(detail_path)
    print("PASS: detail JSON parses")

    # 2. Top-level structure
    for key in ("schema", "provenance", "results", "frames", "aggregate"):
        assert key in detail, f"missing top-level key: {key}"
    print("PASS: top-level structure")

    frames = detail["frames"]
    results = detail["results"]
    rows = detail["aggregate"]["rows"]

    # 3. Counts
    _check("frame count == 30600", len(frames) == EXPECTED_FRAMES, f"got {len(frames)}")
    _check("result count == 4", len(results) == 4, f"got {len(results)}")
    _check("aggregate row count == 4", len(rows) == 4, f"got {len(rows)}")

    # 4. Seeds
    seeds = [r["seed"] for r in results]
    _check("seeds match expected order", seeds == EXPECTED_SEEDS, f"got {seeds}")

    # 5. n_decisions == 7650 per seed
    for r in results:
        _check(
            f"seed {r['seed']}: n_decisions == 7650",
            r["n_decisions"] == EXPECTED_DECISIONS_PER_SEED,
            f"got {r['n_decisions']}",
        )

    # 6. Required frame fields
    required = [
        "seed", "episode", "tick", "agent_index",
        "pre_step_obs95", "action_taken", "probs", "action_mask", "onball",
        "team_has_ball", "agent_has_ball", "pre_step_ball_owner_agent_idx",
    ]
    missing = []
    for fr in frames:
        for k in required:
            if k not in fr:
                missing.append((fr.get("seed"), fr.get("episode"), fr.get("tick"), fr.get("agent_index"), k))
    _check("all frames have required fields", len(missing) == 0, f"missing: {missing[:5]}")

    # 7. No non-finite in probs/masked_logits
    bad = 0
    for fr in frames:
        for p in fr.get("probs", []):
            if isinstance(p, float) and (math.isnan(p) or math.isinf(p)):
                bad += 1
        for m in fr.get("masked_logits", []):
            if isinstance(m, float) and (math.isnan(m) or math.isinf(m)):
                bad += 1
    _check("no non-finite values in probs/masked_logits", bad == 0, f"found {bad}")

    # 8. Checkpoint SHAs
    for r in results:
        sha = r.get("checkpoint_sha256", "")
        _check(f"seed {r['seed']}: checkpoint SHA is 64-char", sha and len(sha) == 64, f"got {sha[:16]}...")

    # 9. CSVs parse
    summary_rows = _load_csv_rows(summary_path)
    recon_rows = _load_csv_rows(recon_path)
    _check("summary CSV has 4 rows", len(summary_rows) == 4, f"got {len(summary_rows)}")
    _check("reconciliation CSV has 4 rows", len(recon_rows) == 4, f"got {len(recon_rows)}")

    # 10. pi_floor passed all seeds
    for row in rows:
        _check(f"seed {row['seed']}: pi_floor passed", row["pi_floor_passed"] is True)

    # 11. Temporal alignment
    temporal_ok = True
    for fr in frames:
        probs = fr["probs"]
        mask = fr["action_mask"]
        action = fr["action_taken"]
        legal = [i for i in range(ACTION_DIM) if mask[i] == 1]
        if not legal:
            temporal_ok = False
            break
        expected = max(legal, key=lambda i: probs[i])
        if action != expected:
            temporal_ok = False
            break
    _check("temporal alignment (action == pre-step masked argmax)", temporal_ok)

    # 12. Semantic invariants (Task 2)
    _validate_semantic_invariants(frames)
    print("PASS: semantic invariants (onball, team_has_ball, agent_has_ball)")

    # 13. Independent reconstruction of summary CSV (Task 4)
    reconstructed_summary = _reconstruct_summary_from_frames(frames)
    _compare_summary_csv(reconstructed_summary, summary_rows)
    print("PASS: summary CSV matches raw-detail reconstruction")

    # 14. Independent reconstruction of reconciliation CSV (Task 3)
    with open("training/results/actor_reweight_retest_eval_summary.csv", "r", encoding="utf-8") as f:
        rebuilt_rates = {int(r["seed"]): float(r["pass_shot_rate_pct"]) for r in csv.DictReader(f)}
    reconstructed_recon = _reconstruct_reconciliation_from_frames(frames, rebuilt_rates)
    _compare_reconciliation_csv(reconstructed_recon, recon_rows)
    print("PASS: reconciliation CSV matches raw-detail reconstruction")

    # 15. Canonical rate matches raw counts (legacy check)
    for row in rows:
        seed = row["seed"]
        n_dec = row["n_decisions"]
        n_ps = row["n_pass_shot"]
        expected_rate = (n_ps / max(1, n_dec)) * 100.0
        actual_rate = row["canonical_rate_pct"]
        _check(
            f"seed {seed}: canonical rate matches raw counts",
            abs(expected_rate - actual_rate) < 1e-9,
            f"expected={expected_rate:.6f} actual={actual_rate:.6f}",
        )

    # 16. Rebuilt CSV comparison
    for row in rows:
        seed = row["seed"]
        rebuilt_rate = rebuilt_rates.get(seed)
        if rebuilt_rate is not None:
            actual_rate = row["canonical_rate_pct"]
            delta = abs(actual_rate - rebuilt_rate)
            _check(
                f"seed {seed}: canonical rate within {RECONCILIATION_TOLERANCE_PP}pp of rebuilt CSV",
                delta <= RECONCILIATION_TOLERANCE_PP,
                f"delta={delta:.6f}pp",
            )

    print()
    print("ALL VERIFICATION CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
