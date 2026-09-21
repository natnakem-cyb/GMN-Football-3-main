"""Compute canonical pi statistics from existing detail JSON."""
import json
from collections import defaultdict

PASS_ACTION_IDS = (9, 10, 11)
SHOT_ACTION_IDS = (12,)
PASS_SHOT_ACTION_IDS = (9, 10, 11, 12)
MOVE_ACTION_IDS = (1, 2, 3, 4, 5, 6, 7, 8)
ACTION_DIM = 19


def load_detail(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def compute_stats(seed_frames):
    onball = [f for f in seed_frames if f.get("onball")]
    n_onball = len(onball)

    n_pass_legal = sum(1 for f in onball if f.get("mask_pass_legal") == 1)
    n_shot_legal = sum(1 for f in onball if f.get("mask_shot_legal") == 1)
    n_either_legal = sum(1 for f in onball if f.get("mask_pass_or_shot_legal") == 1)

    pi_pass_all = [f["pi_pass"] for f in onball]
    pi_shot_all = [f["pi_shot"] for f in onball]
    pi_ps_all = [f["pi_pass_shot"] for f in onball]
    entropy_all = [f["entropy"] for f in onball]

    pi_pass_pass_legal = [f["pi_pass"] for f in onball if f.get("mask_pass_legal") == 1]
    pi_shot_shot_legal = [f["pi_shot"] for f in onball if f.get("mask_shot_legal") == 1]
    pi_ps_either_legal = [f["pi_pass_shot"] for f in onball if f.get("mask_pass_or_shot_legal") == 1]

    delta_pass_move_list = []
    delta_shot_move_list = []
    for f in onball:
        masked_logits = f.get("masked_logits", [])
        if len(masked_logits) != ACTION_DIM:
            continue
        action_mask = f.get("action_mask", [])
        pass_logits = [masked_logits[i] for i in PASS_ACTION_IDS if action_mask[i] == 1]
        shot_logits = [masked_logits[i] for i in SHOT_ACTION_IDS if action_mask[i] == 1]
        move_logits = [masked_logits[i] for i in MOVE_ACTION_IDS if action_mask[i] == 1]
        if pass_logits and move_logits:
            delta_pass_move_list.append(max(pass_logits) - max(move_logits))
        if shot_logits and move_logits:
            delta_shot_move_list.append(max(shot_logits) - max(move_logits))

    n_selected_ps = sum(1 for f in onball if f.get("action_taken") in PASS_SHOT_ACTION_IDS)
    selected_frames = [f for f in onball if f.get("action_taken") in PASS_SHOT_ACTION_IDS]

    if selected_frames:
        pi_selected_values = [f["probs"][f["action_taken"]] for f in selected_frames]
        pi_ps_selected_values = [f["pi_pass_shot"] for f in selected_frames]
        mask_sum_selected = [f.get("mask_sum", 0) for f in selected_frames]
        mean_pi_selected = sum(pi_selected_values) / len(pi_selected_values)
        mean_pi_ps_given_selected = sum(pi_ps_selected_values) / len(pi_ps_selected_values)
        mean_mask_sum_given_selected = sum(mask_sum_selected) / len(mask_sum_selected)
    else:
        mean_pi_selected = None
        mean_pi_ps_given_selected = None
        mean_mask_sum_given_selected = None

    if n_selected_ps > 0 and n_onball > 0:
        floor_action_space = (n_selected_ps / n_onball) * (1.0 / ACTION_DIM)
    else:
        floor_action_space = None

    if selected_frames:
        floor_observed_masks = sum(1.0 / f.get("mask_sum", 1) for f in selected_frames) / n_onball
    else:
        floor_observed_masks = None

    mean_pi_ps = sum(pi_ps_all) / max(1, n_onball)
    if mean_pi_selected is not None and n_onball > 0:
        lhs = n_selected_ps * mean_pi_selected / n_onball
        ineq_pass = lhs <= mean_pi_ps + 1e-9
    else:
        ineq_pass = None

    return {
        "n_onball": n_onball,
        "n_pass_legal": n_pass_legal,
        "n_shot_legal": n_shot_legal,
        "n_either_legal": n_either_legal,
        "pi_pass_mean_pass_legal": sum(pi_pass_pass_legal) / max(1, len(pi_pass_pass_legal)) if pi_pass_pass_legal else None,
        "pi_shot_mean_shot_legal": sum(pi_shot_shot_legal) / max(1, len(pi_shot_shot_legal)) if pi_shot_shot_legal else None,
        "pi_pass_shot_mean_either_legal": sum(pi_ps_either_legal) / max(1, len(pi_ps_either_legal)) if pi_ps_either_legal else None,
        "pi_pass_mean_all_onball": sum(pi_pass_all) / max(1, n_onball),
        "pi_shot_mean_all_onball": sum(pi_shot_all) / max(1, n_onball),
        "pi_pass_shot_mean_all_onball": sum(pi_ps_all) / max(1, n_onball),
        "entropy_mean": sum(entropy_all) / max(1, n_onball),
        "delta_pass_move_mean": sum(delta_pass_move_list) / max(1, len(delta_pass_move_list)) if delta_pass_move_list else None,
        "delta_shot_move_mean": sum(delta_shot_move_list) / max(1, len(delta_shot_move_list)) if delta_shot_move_list else None,
        "n_selected_ps": n_selected_ps,
        "p_selected_ps_given_onball": n_selected_ps / max(1, n_onball),
        "mean_pi_selected": mean_pi_selected,
        "mean_pi_ps_given_selected": mean_pi_ps_given_selected,
        "mean_mask_sum_given_selected": mean_mask_sum_given_selected,
        "floor_action_space": floor_action_space,
        "floor_observed_masks": floor_observed_masks,
        "inequality_pass": ineq_pass,
    }


def main():
    detail_path = "training/results/post_reweight_logit_prestep_reconciled_detail.json"
    detail = load_detail(detail_path)
    frames = detail["frames"]

    by_seed = defaultdict(list)
    for fr in frames:
        by_seed[int(fr["seed"])].append(fr)

    print("CANONICAL PI STATISTICS (from existing detail JSON)")
    print("=" * 100)
    print()
    print("Protocol: base_seed=500000, individual-carrier onball, scenario=academy_3_vs_1_with_keeper_onball, 50 ep/seed")
    print()

    print("TASK 2 — CANONICAL PI STATISTICS")
    print("-" * 100)
    print()
    print("PI STATISTICS (on-ball frames only)")
    print("Seed | n_onball | n_PASS_legal | n_SHOT_legal | n_either_legal | pi_PASS_mean | pi_SHOT_mean | pi_PASS+SHOT_mean | H(pi) | delta_PASS-MOVE | delta_SHOT-MOVE")
    for seed in [42, 123, 7, 999]:
        stats = compute_stats(by_seed[seed])
        print(f"{seed:4d} | {stats['n_onball']:8d} | {stats['n_pass_legal']:11d} | {stats['n_shot_legal']:11d} | {stats['n_either_legal']:13d} | "
              f"{stats['pi_pass_mean_all_onball']:.6f} | {stats['pi_shot_mean_all_onball']:.6f} | {stats['pi_pass_shot_mean_all_onball']:.6f} | "
              f"{stats['entropy_mean']:.6f} | {stats['delta_pass_move_mean']:.6f} | {stats['delta_shot_move_mean']:.6f}")

    print()
    print("LEGAL-CONDITIONAL PI STATISTICS")
    print("Seed | n_onball | n_PASS_legal | pi_PASS_mean (legal) | n_SHOT_legal | pi_SHOT_mean (legal) | n_either_legal | pi_PASS+SHOT_mean (legal)")
    for seed in [42, 123, 7, 999]:
        stats = compute_stats(by_seed[seed])
        print(f"{seed:4d} | {stats['n_onball']:8d} | {stats['n_pass_legal']:11d} | {stats['pi_pass_mean_pass_legal']:.6f} | {stats['n_shot_legal']:11d} | "
              f"{stats['pi_shot_mean_shot_legal']:.6f} | {stats['n_either_legal']:13d} | {stats['pi_pass_shot_mean_either_legal']:.6f}")

    print()
    print("DETERMINISTIC SELECTION FREQUENCY (separate from pi statistics)")
    print("Seed | n_onball | n_selected_PS | P(selected PS | on-ball, legal)")
    for seed in [42, 123, 7, 999]:
        stats = compute_stats(by_seed[seed])
        print(f"{seed:4d} | {stats['n_onball']:8d} | {stats['n_selected_ps']:13d} | {stats['p_selected_ps_given_onball']:.6f}")

    print()
    print("TASK 3 — PI-FLOOR RECONCILIATION")
    print("-" * 100)
    print()
    print("Seed | n_onball | n_selected_PS | mean_pi_PS | mean(pi_selected) | mean(mask_sum|selected) | floor_action | floor_mask | ineq_pass")
    for seed in [42, 123, 7, 999]:
        stats = compute_stats(by_seed[seed])
        print(f"{seed:4d} | {stats['n_onball']:8d} | {stats['n_selected_ps']:13d} | "
              f"{stats['pi_pass_shot_mean_all_onball']:.6f} | {stats['mean_pi_selected']:.6f} | {stats['mean_mask_sum_given_selected']:.6f} | "
              f"{stats['floor_action_space']:.6f} | {stats['floor_observed_masks']:.6f} | {stats['inequality_pass']}")

    print()
    all_pass = all(compute_stats(by_seed[seed])["inequality_pass"] is not False for seed in [42, 123, 7, 999])
    print(f"All seeds pass pi-floor: {all_pass}")

    print()
    print("TASK 4 — SEED-42 DISCONNECT RE-EXAMINATION")
    print("-" * 100)
    print()

    pi_ps_means = {}
    sel_freqs = {}
    n_onballs = {}
    for seed in [42, 123, 7, 999]:
        stats = compute_stats(by_seed[seed])
        pi_ps_means[seed] = stats["pi_pass_shot_mean_all_onball"]
        sel_freqs[seed] = stats["p_selected_ps_given_onball"]
        n_onballs[seed] = stats["n_onball"]

    print("Seed | n_onball | pi_PASS+SHOT_mean | P(selected PS | on-ball) | Unconditional rate")
    rates = {42: 0.745098, 123: 1.673203, 7: 0.862745, 999: 1.032680}
    for seed in [42, 123, 7, 999]:
        print(f"{seed:4d} | {n_onballs[seed]:8d} | {pi_ps_means[seed]:.6f} | {sel_freqs[seed]:.6f} | {rates[seed]:.6f}%")

    print()
    print(f"Seed 42 small-n caveat: 1 frame = {1.0/n_onballs[42]*100:.2f} pp")
    print()

    print("Occupancy arithmetic (all-agent scope):")
    print("Seed | P(on-ball) | P(selected PS | on-ball) | Implied unconditional rate | Actual unconditional rate | Match?")
    for seed in [42, 123, 7, 999]:
        stats = compute_stats(by_seed[seed])
        p_onball = stats["n_onball"] / 7650.0
        p_sel = stats["p_selected_ps_given_onball"]
        implied = p_onball * p_sel * 100.0
        actual = rates[seed]
        match = abs(implied - actual) < 0.01
        print(f"{seed:4d} | {p_onball:.6f} | {p_sel:.6f} | {implied:.6f}% | {actual:.6f}% | {'YES' if match else 'NO'}")

    print()
    print("Note: The implied unconditional rate uses the ALL-AGENT denominator (7650).")
    print("This is the correct scope for comparing against the canonical unconditional rate.")


if __name__ == "__main__":
    main()
