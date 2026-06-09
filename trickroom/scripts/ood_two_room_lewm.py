"""Trick Room — LeWM OOD eval across Two-Room factors of variation.

Runs the same 8 regimes as ood_two_room.py but with the pretrained LeWM
model (quentinll/lewm-tworooms) instead of the symbolic substrate.

LeWM was trained on 1000 trajectories of the DEFAULT regime.  Here we ask
how it generalises to distribution-shifted envs without retraining.  The
substrate gets 100% flat across all regimes (both privileged and fixed-prior)
as shown by ood_two_room_substrate.json; this script produces the contrast.

Runtime estimate on MPS: ~25–35 min/regime → ~4–5 hours total.
"""

from __future__ import annotations

import argparse
import json
import time
import warnings
from datetime import datetime
from pathlib import Path

warnings.filterwarnings('ignore')

import numpy as np
import stable_pretraining as spt  # noqa: F401 (must import before swm on some builds)
import torch
from torchvision.transforms import v2 as transforms

import stable_worldmodel as swm
from stable_worldmodel.data.normalization import IdentityScaler
from stable_worldmodel.policy import PlanConfig, WorldModelPolicy
from stable_worldmodel.solver import CEMSolver

_IMAGENET = {'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]}

# Identical regime table as ood_two_room.py — env_init values only;
# LeWM has no substrate_kwargs because it has no prior to update.
REGIMES = [
    ('default',          {}),
    ('wall_horizontal',  {'wall.axis': 0}),
    ('wall_thick_20',    {'wall.thickness': 20}),
    ('agent_fast_8',     {'agent.speed': np.array([8.0], dtype=np.float32)}),
    ('agent_slow_3',     {'agent.speed': np.array([3.0], dtype=np.float32)}),
    ('agent_big_10',     {'agent.radius': np.array([10.0], dtype=np.float32)}),
    ('door_big_20',      {'door.size': np.array([20, 14, 14], dtype=int)}),
    ('three_doors',      {'door.number': 3,
                          'door.position': np.array([50, 112, 175], dtype=int),
                          'door.size':     np.array([14, 14, 14], dtype=int)}),
]


def _img_transform(dtype=torch.float32):
    return transforms.Compose([
        transforms.ToImage(),
        transforms.ToDtype(dtype, scale=True),
        transforms.Normalize(**_IMAGENET),
        transforms.Resize(size=224),
    ])


def _build_policy(model, device: str):
    transform = {'pixels': _img_transform(), 'goal': _img_transform()}
    process   = {'action': IdentityScaler(), 'proprio': IdentityScaler()}
    solver = CEMSolver(
        model=model, num_samples=300, n_steps=30, topk=30,
        batch_size=25, device=device,
    )
    config = PlanConfig(horizon=5, receding_horizon=5, action_block=5)
    return WorldModelPolicy(
        solver=solver, config=config, process=process, transform=transform,
    )


def run_regime(label: str, env_init: dict, model, device: str,
               num_eval: int = 50, seed: int = 42) -> dict:
    world_kwargs: dict = {
        'num_envs': num_eval,
        'image_shape': (224, 224),
        'max_episode_steps': 100,
    }
    if env_init:
        world_kwargs['init_value'] = env_init

    world = swm.World('swm/TwoRoom-v1', **world_kwargs)
    policy = _build_policy(model, device)
    world.set_policy(policy)

    t0 = time.time()
    results = world.evaluate(episodes=num_eval, seed=seed)
    elapsed = time.time() - t0

    env_init_serialisable = {
        k: v.tolist() if isinstance(v, np.ndarray) else v
        for k, v in env_init.items()
    }

    return {
        'label': label,
        'env_init_value': env_init_serialisable,
        'success_rate_pct': float(results['success_rate']),
        'episodes_solved': int(results['episode_successes'].sum()),
        'episodes_total': num_eval,
        'wall_clock_s': elapsed,
        'wall_clock_per_episode_ms': elapsed / num_eval * 1000.0,
    }


def _print_table(rows: list[dict]) -> None:
    print('=' * 72)
    print(' TRICK ROOM OOD TABLE — LeWM (pretrained, no fine-tuning)')
    print('=' * 72)
    print(f'{"regime":<22} {"success":>10} {"solved":>10} {"ms/ep":>10}')
    for r in rows:
        if 'error' in r:
            print(f'{r["label"]:<22} ERROR: {r["error"][:60]}')
        else:
            print(
                f'{r["label"]:<22} {r["success_rate_pct"]:>9.1f}%'
                f' {r["episodes_solved"]:>3}/{r["episodes_total"]:<6}'
                f' {r["wall_clock_per_episode_ms"]:>9.1f}'
            )
    print('=' * 72)


def main(num_eval: int = 50, seed: int = 42,
         out: str = 'results/ood_two_room_lewm.json',
         regimes: list[str] | None = None) -> dict:

    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f'[LeWM OOD] device: {device}')

    model = swm.wm.utils.load_pretrained('quentinll/lewm-tworooms')
    model = model.to(device).eval()
    model.requires_grad_(False)
    model.interpolate_pos_encoding = True
    n_params = sum(p.numel() for p in model.parameters())
    print(f'[LeWM OOD] {n_params:,} params loaded')

    regime_filter = set(regimes) if regimes else None

    rows: list[dict] = []
    for label, env_init in REGIMES:
        if regime_filter and label not in regime_filter:
            continue
        print(f'[LeWM OOD] running {label}...')
        try:
            row = run_regime(label, env_init, model, device,
                             num_eval=num_eval, seed=seed)
        except Exception as e:
            env_init_serialisable = {
                k: v.tolist() if isinstance(v, np.ndarray) else v
                for k, v in env_init.items()
            }
            row = {'label': label, 'error': repr(e),
                   'env_init_value': env_init_serialisable}
        rows.append(row)
        _sr = row.get('success_rate_pct', 'ERR')
        print(f'[LeWM OOD]   → {label}: {_sr}%')

    summary = {
        'label': 'lewm_ood',
        'env': 'swm/TwoRoom-v1',
        'checkpoint': 'quentinll/lewm-tworooms',
        'device': device,
        'config': {
            'num_eval': num_eval,
            'seed': seed,
            'cem': {'num_samples': 300, 'n_steps': 30, 'topk': 30},
            'plan': {'horizon': 5, 'receding_horizon': 5, 'action_block': 5},
        },
        'training_trajectories': 1000,
        'regimes': rows,
        'timestamp': datetime.utcnow().isoformat() + 'Z',
    }

    out_path = Path(__file__).resolve().parents[2] / out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2) + '\n')

    _print_table(rows)
    print(f'saved to {out_path}')
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-eval', type=int, default=50)
    parser.add_argument('--seed',     type=int, default=42)
    parser.add_argument('--out',      type=str,
                        default='results/ood_two_room_lewm.json')
    parser.add_argument('--regimes',  type=str, nargs='*', default=None,
                        help='Run a subset of regimes (space-separated names)')
    args = parser.parse_args()
    main(num_eval=args.num_eval, seed=args.seed, out=args.out,
         regimes=args.regimes)
