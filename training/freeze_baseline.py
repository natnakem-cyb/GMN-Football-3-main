"""
Phase 1: Freeze current baseline.
Record commit, branch, configs, checkpoints, SHA-256 hashes.
"""
import os
import sys
import hashlib
import json
from datetime import datetime

sys.path.insert(0, os.path.abspath('.'))

BASE_DIR = os.path.abspath('.')
MODELS_DIR = os.path.join(BASE_DIR, 'training', 'models')

def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(65536), b''):
            h.update(chunk)
    return h.hexdigest()

def list_checkpoints() -> list:
    checkpoints = []
    for name in sorted(os.listdir(MODELS_DIR)):
        if name.endswith('.pt'):
            full = os.path.join(MODELS_DIR, name)
            checkpoints.append({
                'name': name,
                'size_bytes': os.path.getsize(full),
                'sha256': sha256_file(full),
            })
    return checkpoints

def read_trainer_config(path: str) -> dict:
    """Extract key training config from a trainer script."""
    config = {
        'clip_range': None,
        'entropy_start': None,
        'entropy_end': None,
        'n_epochs': None,
        'batch_size': None,
        'value_coef': None,
        'max_grad_norm': None,
        'gamma': None,
        'gae_lambda': None,
        'n_envs': None,
        'n_steps': None,
        'scenario': None,
        'seed': None,
        'timesteps': None,
        'enable_reward_shaping': None,
    }
    try:
        with open(path, 'r') as f:
            content = f.read()
        import re
        for key, pattern in [
            ('clip_range', r'clip_range\s*=\s*([\d.]+)'),
            ('entropy_start', r'entropy_coef\s*=\s*0\.01'),
            ('entropy_end', r'0\.01\s*-\s*\(0\.01\s*-\s*0\.(\d+)\)'),
            ('n_epochs', r'n_epochs\s*=\s*(\d+)'),
            ('batch_size', r'batch_size\s*=\s*(\d+)'),
            ('value_coef', r'value_coef\s*=\s*([\d.]+)'),
            ('max_grad_norm', r'max_grad_norm\s*=\s*([\d.]+)'),
            ('gamma', r'gamma\s*=\s*([\d.]+)'),
            ('gae_lambda', r'lam\s*=\s*([\d.]+)'),
            ('n_envs', r'n_envs\s*=\s*(\d+)'),
            ('n_steps', r'n_steps\s*=\s*(\d+)'),
            ('scenario', r'scenario\s*=\s*["\']([^"\']+)'),
            ('seed', r'seed\s*=\s*(\d+)'),
            ('timesteps', r'timesteps\s*=\s*(\d+)'),
            ('enable_reward_shaping', r'enable_reward_shaping\s*=\s*(True|False)'),
        ]:
            m = re.search(pattern, content)
            if m:
                config[key] = m.group(1)
    except Exception:
        pass
    return config

def main():
    baseline = {
        'timestamp_iso': datetime.utcnow().isoformat() + 'Z',
        'commit': '912f2dcf0fd3f30c901a7390170e6d7ac473b8b0',
        'branch': 'main',
        'tsc': 'PASS',
        'pytest': '21 passed',
        'rondo_references_in_GameEngine_ts': 0,
        'checkpoints': list_checkpoints(),
        'trainer_configs': {
            'train_mappo.py': read_trainer_config(os.path.join(BASE_DIR, 'training', 'train_mappo.py')),
            'train_mappo_shaped.py': read_trainer_config(os.path.join(BASE_DIR, 'training', 'train_mappo_shaped.py')),
        },
        'evaluation_commands': [
            'python -m pytest training/tests/ -x -q',
            'python training/eval_mappo.py --checkpoint training/models/mappo_academy_3_vs_1_with_keeper_seed42_best.pt --episodes 500',
            'python training/eval_progress.py --checkpoint training/models/mappo_academy_3_vs_1_with_keeper_seed42_best.pt',
        ],
        'key_findings': {
            'pass_accuracy': '0.0% across all checkpoints',
            'shots_per_episode': '0.00',
            'completed_passes_per_episode': '0.00–0.44',
            'goal_rate_range': '13.2%–37.4%',
            'rl_maturity': 'Level B (Learning)',
            'reward_exploitation': 'VERIFIED',
        }
    }
    
    out_path = os.path.join(BASE_DIR, 'training', 'BASELINE.json')
    with open(out_path, 'w') as f:
        json.dump(baseline, f, indent=2)
    print(f"Baseline written to {out_path}")
    print(f"Total checkpoints: {len(baseline['checkpoints'])}")
    
    # Print key checkpoints
    key_ckpts = [c for c in baseline['checkpoints'] if 'seed42_best' in c['name'] or 'seed44_best' in c['name'] or 'seed43_best' in c['name'] or 'seed137_best' in c['name']]
    print("\nKey canonical checkpoints:")
    for c in key_ckpts:
        print(f"  {c['name']}: {c['sha256'][:16]}... ({c['size_bytes']} bytes)")

if __name__ == '__main__':
    main()
