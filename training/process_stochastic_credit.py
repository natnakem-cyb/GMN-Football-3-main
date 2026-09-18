"""
Process stochastic rollout JSONs into per-seed event-centered credit measurements.
Outputs CSV with V, δ, A stats by event type and seed.
"""
import json, csv, statistics, os, glob

RESULTS_DIR = "training/results"
OUT_CSV = os.path.join(RESULTS_DIR, "stochastic_rollout_credit_summary.csv")

FILES = sorted(glob.glob(os.path.join(RESULTS_DIR, "stochastic_rollout_*_freshtrain_50176.json")))

rows = []

for path in FILES:
    with open(path, "r") as f:
        data = json.load(f)

    ckpt = data["checkpoint"]
    seed_label = ckpt.split("_seed")[1].split("_")[0]
    episodes = data["episodes"]
    total_ticks = 0
    # Aggregate across all episodes and all agents
    stats = {}
    for ep in episodes:
        for tick in ep.get("tick_log", []):
            total_ticks += 1
            etype = tick.get("event_type")
            if etype is None:
                continue
            if etype not in stats:
                stats[etype] = {"V": [], "delta": [], "A": []}
            stats[etype]["V"].append(tick.get("value", 0.0))
            stats[etype]["delta"].append(tick.get("td_residual", 0.0))
            stats[etype]["A"].append(tick.get("gae_advantage", 0.0))

    for etype, vals in stats.items():
        n = len(vals["V"])
        if n == 0:
            continue
        rows.append({
            "seed": seed_label,
            "event_type": etype,
            "n": n,
            "V_mean": statistics.mean(vals["V"]),
            "V_std": statistics.stdev(vals["V"]) if n > 1 else 0.0,
            "delta_mean": statistics.mean(vals["delta"]),
            "delta_std": statistics.stdev(vals["delta"]) if n > 1 else 0.0,
            "A_mean": statistics.mean(vals["A"]),
            "A_std": statistics.stdev(vals["A"]) if n > 1 else 0.0,
        })

# Write CSV
fieldnames = ["seed", "event_type", "n", "V_mean", "V_std", "delta_mean", "delta_std", "A_mean", "A_std"]
with open(OUT_CSV, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

print(f"Wrote {len(rows)} rows to {OUT_CSV}")
