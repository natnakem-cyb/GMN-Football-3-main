"""Independent verification for canonical 3-agent reconciled artifacts."""
import csv
import json
import math


def main() -> int:
    detail_path = "training/results/post_reweight_logit_prestep_reconciled_detail.json"
    summary_path = "training/results/post_reweight_logit_prestep_reconciled_summary.csv"
    recon_path = "training/results/post_reweight_canonical_scope_reconciliation.csv"

    print("=== INDEPENDENT VERIFICATION: CANONICAL 3-AGENT ARTIFACTS ===")

    # 1. JSON parses
    with open(detail_path, "r", encoding="utf-8") as f:
        text = f.read()
    detail = json.loads(text)
    print("PASS: detail JSON parses")

    # 2. Structure
    for key in ("schema", "provenance", "results", "frames", "aggregate"):
        assert key in detail, f"missing top-level key: {key}"
    print("PASS: top-level structure")

    frames = detail["frames"]
    results = detail["results"]
    rows = detail["aggregate"]["rows"]

    # 3. Counts
    assert len(frames) == 30600, f"expected 30600 frames, got {len(frames)}"
    assert len(results) == 4, f"expected 4 results, got {len(results)}"
    assert len(rows) == 4, f"expected 4 rows, got {len(rows)}"
    print("PASS: frame/result/row counts")

    # 4. Seeds
    seeds = [r["seed"] for r in results]
    assert seeds == [42, 123, 7, 999], f"wrong seeds: {seeds}"
    print("PASS: all 4 seeds present")

    # 5. n_decisions == 7650
    for r in results:
        assert r["n_decisions"] == 7650, f"seed {r['seed']} n_decisions={r['n_decisions']}"
    print("PASS: all seeds have 7650 decisions")

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
    assert not missing, f"missing fields: {missing[:5]}"
    print("PASS: all frames have required fields")

    # 7. No non-finite in probs/masked_logits
    bad = 0
    for fr in frames:
        for p in fr.get("probs", []):
            if isinstance(p, float) and (math.isnan(p) or math.isinf(p)):
                bad += 1
        for m in fr.get("masked_logits", []):
            if isinstance(m, float) and (math.isnan(m) or math.isinf(m)):
                bad += 1
    assert bad == 0, f"found {bad} non-finite values"
    print("PASS: no non-finite values in probs/masked_logits")

    # 8. Checkpoint SHAs
    for r in results:
        sha = r.get("checkpoint_sha256", "")
        assert sha and len(sha) == 64, f"bad sha for seed {r['seed']}"
    print("PASS: all checkpoint SHAs present and 64-char")

    # 9. CSVs parse
    for name in (summary_path, recon_path):
        with open(name, "r", encoding="utf-8") as f:
            csv_rows = list(csv.DictReader(f))
        assert len(csv_rows) == 4, f"{name} has {len(csv_rows)} rows"
    print("PASS: both CSVs parse with 4 rows")

    # 10. pi_floor passed all seeds
    for row in rows:
        assert row["pi_floor_passed"] is True, f"seed {row['seed']} pi_floor failed"
    print("PASS: all seeds pass pi_floor")

    # 11. Temporal alignment
    temporal_ok = True
    for fr in frames:
        probs = fr["probs"]
        mask = fr["action_mask"]
        action = fr["action_taken"]
        legal = [i for i in range(19) if mask[i] == 1]
        if not legal:
            temporal_ok = False
            break
        expected = max(legal, key=lambda i: probs[i])
        if action != expected:
            temporal_ok = False
            break
    assert temporal_ok, "temporal alignment failed"
    print("PASS: temporal alignment (action == pre-step masked argmax)")

    # 12. Canonical rate matches raw counts
    for row in rows:
        seed = row["seed"]
        n_dec = row["n_decisions"]
        n_ps = row["n_pass_shot"]
        expected_rate = (n_ps / max(1, n_dec)) * 100.0
        actual_rate = row["canonical_rate_pct"]
        assert abs(expected_rate - actual_rate) < 1e-9, f"seed {seed} rate mismatch"
    print("PASS: canonical rate matches raw counts")

    # 13. Rebuilt CSV comparison
    with open("training/results/actor_reweight_retest_eval_summary.csv", "r", encoding="utf-8") as f:
        rebuilt = {int(r["seed"]): float(r["pass_shot_rate_pct"]) for r in csv.DictReader(f)}
    for row in rows:
        seed = row["seed"]
        rebuilt_rate = rebuilt.get(seed)
        if rebuilt_rate is not None:
            actual_rate = row["canonical_rate_pct"]
            delta = abs(actual_rate - rebuilt_rate)
            # Allow 0.5pp tolerance for measurement variance
            assert delta <= 0.5, f"seed {seed} delta {delta:.3f}pp exceeds tolerance"
    print("PASS: canonical rates within tolerance of rebuilt CSV")

    print()
    print("ALL VERIFICATION CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
