# Phase 14 — Generalization Protocol

## Changed Initial Conditions

Vary:
1. Player positions (use scenario positionJitter > 0.05)
2. Ball position (offset from default)
3. Defender position (shifted left/right)
4. Random initialization (different seeds)
5. Scenario seed (different episode seeds)

## Procedure

For each variation:
1. Load best checkpoint from primary training
2. Run 30 deterministic episodes
3. Compare metrics against training distribution

## Variations to Test

### academy_3_vs_1_with_keeper
- Default (positionJitter=0.05)
- High jitter (positionJitter=0.12)
- Shifted defender (custom scenario with CB at x=0.42)
- Randomized ball start (ball x in [0.15, 0.35])

### academy_3_vs_1_defender_2
- 3v2 with extra defender
- Tests policy robustness against defensive overload

### academy_3_vs_1_shifted
- Asymmetric defender positioning
- Tests policy adaptability

## Success Criterion

Behavior generalizes if:
- Pass accuracy > 3% in all variations
- Goal rate > 15% in all variations
- No catastrophic collapse (goal rate > 5% in all variations)
