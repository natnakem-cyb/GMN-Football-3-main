import hashlib
import os

files = [
    "training/models/mappo_academy_3_vs_1_with_keeper_best.pt",
    "training/models/mappo_academy_3_vs_1_with_keeper_seed43.pt",
    "training/models/mappo_academy_3_vs_1_with_keeper_seed44.pt",
    "training/models/mappo_academy_3_vs_1_with_keeper_seed137.pt",
]

for f in files:
    if os.path.exists(f):
        h = hashlib.sha256(open(f, "rb").read()).hexdigest()
        print(f"{f}: {h}")
    else:
        print(f"{f}: MISSING")
