"""Trick Room v0 — LeWM baseline on Two-Room.

Loads quentinll/lewm-tworooms (official pretrained checkpoint from upstream
co-author Quentin Le Lidec). No training needed from our side.  Runs the
same CEM-MPC config as the substrate headline:
  300 samples × 30 iters, horizon 5, receding 5, action_block 5, seed 42.

No 3.4 GB dataset download needed:
  - LeWM's get_cost uses pixels + goal only; proprio is ignored.
  - CEM operates in raw action space (N(0,1) candidates get env-clipped to
    [-1, 1] by TwoRoom's action space — IdentityScaler is exact here).
"""

from __future__ import annotations

import argparse
import json
import time
import warnings
from datetime import datetime
from pathlib import Path

warnings.filterwarnings('ignore')

import stable_pretraining as spt
import torch
from torchvision.transforms import v2 as transforms

import stable_worldmodel as swm
from stable_worldmodel.data.normalization import IdentityScaler
from stable_worldmodel.policy import PlanConfig, WorldModelPolicy
from stable_worldmodel.solver import CEMSolver


_IMAGENET = {'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]}


def _img_transform(dtype=torch.float32):
    return transforms.Compose([
        transforms.ToImage(),
        transforms.ToDtype(dtype, scale=True),
        transforms.Normalize(**_IMAGENET),
        transforms.Resize(size=224),
    ])


def main(num_eval: int = 50, seed: int = 42,
         out: str = 'results/lewm_two_room.json'):

    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f'[LeWM] device: {device}')

    model = swm.wm.utils.load_pretrained('quentinll/lewm-tworooms')
    model = model.to(device).eval()
    model.requires_grad_(False)
    model.interpolate_pos_encoding = True
    n_params = sum(p.numel() for p in model.parameters())
    print(f'[LeWM] {n_params:,} params loaded')

    transform = {
        'pixels': _img_transform(),
        'goal': _img_transform(),
    }
    process = {
        'action': IdentityScaler(),
        'proprio': IdentityScaler(),
    }

    solver = CEMSolver(
        model=model,
        num_samples=300,
        n_steps=30,
        topk=30,
        batch_size=25,
        device=device,
    )
    config = PlanConfig(horizon=5, receding_horizon=5, action_block=5)
    policy = WorldModelPolicy(
        solver=solver, config=config, process=process, transform=transform,
    )

    world = swm.World(
        'swm/TwoRoom-v1',
        num_envs=num_eval,
        image_shape=(224, 224),
        max_episode_steps=100,
    )
    world.set_policy(policy)

    print(f'[LeWM] running {num_eval} episodes (seed={seed})...')
    t0 = time.time()
    results = world.evaluate(episodes=num_eval, seed=seed)
    elapsed = time.time() - t0

    summary = {
        'label': 'lewm_quentinll_pretrained',
        'env': 'swm/TwoRoom-v1',
        'checkpoint': 'quentinll/lewm-tworooms',
        'device': device,
        'config': {
            'num_eval': num_eval,
            'seed': seed,
            'cem': {'num_samples': 300, 'n_steps': 30, 'topk': 30},
            'plan': {'horizon': 5, 'receding_horizon': 5, 'action_block': 5},
        },
        'success_rate_pct': float(results['success_rate']),
        'episodes_solved': int(results['episode_successes'].sum()),
        'episodes_total': num_eval,
        'wall_clock_total_s': elapsed,
        'wall_clock_per_episode_ms': elapsed / num_eval * 1000.0,
        'training_trajectories': 1000,
        'timestamp': datetime.utcnow().isoformat() + 'Z',
    }

    out_path = Path(__file__).resolve().parents[2] / out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2) + '\n')

    print('=' * 64)
    print(' TRICK ROOM — LeWM BASELINE (Two-Room)')
    print('=' * 64)
    print(f'success_rate: {summary["success_rate_pct"]:.1f}%')
    print(f'episodes:     {summary["episodes_solved"]}/{summary["episodes_total"]}')
    print(f'ms/episode:   {summary["wall_clock_per_episode_ms"]:.1f}')
    print(f'training:     {summary["training_trajectories"]} trajectories')
    print('=' * 64)
    print(f'saved to {out_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-eval', type=int, default=50)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--out', type=str, default='results/lewm_two_room.json')
    args = parser.parse_args()
    main(num_eval=args.num_eval, seed=args.seed, out=args.out)
