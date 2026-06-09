"""Trick Room — Harder OOD stress test for the substrate (fixed_prior).

Standard ood_two_room_pyo3.py leaves the substrate at ~96-100% across all 8
regimes because geometric rescue dominates:
  - 29/50 episodes are same-room (no wall crossing needed)
  - Default door at env y=49, substrate prior y=112: 63px mismatch, yet CEM
    accidentally navigates close enough to the real door on most cross-room eps

This script breaks geometric rescue by moving the env door far from the
substrate's assumed position (y=112) and confirming that cross-room episodes
actually fail.  Predicted outcomes:

    door_default_49    ~96-100%   (63px mismatch, geometric rescue holds)
    door_at_180        ~58-70%    (68px mismatch, rescue weaker at extremes)
    door_at_30         ~58-70%    (82px mismatch, symmetric to door_at_180)
    door_at_200        ~58%       (88px mismatch, near maximum; floor = same-room)

The ~58% floor comes from the 29/50 same-room episodes that need no crossing.
Any result substantially above 58% is residual geometric luck; any result near
58% is the substrate actually failing cross-room episodes.

Run:
    cd stable-worldmodel-trickroom
    python scripts/benchmark/ood_harder_substrate.py
"""

from __future__ import annotations

import argparse
import json
import time
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np

warnings.filterwarnings('ignore')

import stable_worldmodel as swm
from stable_worldmodel.policy import PlanConfig, WorldModelPolicy
from stable_worldmodel.solver import CEMSolver
from stable_worldmodel.wm.substrate import Pyo3SubstrateCostModel


# door.position space: MultiDiscrete(nvec=[224,224,224])
# All three slots must be filled; door.number controls how many are active.
# Substrate fixed prior: door_positions=[112.0] (hardcoded default in pyo3_model.py).
# The gap between substrate prior and env door creates the failure signal.
HARDER_REGIMES = [
    (
        'door_default_49',
        {'door.position': np.array([49, 49, 49], dtype=int), 'door.number': 1},
        'env door y=49 (env default), substrate prior y=112 (63px mismatch)',
    ),
    (
        'door_at_180',
        {'door.position': np.array([180, 49, 49], dtype=int), 'door.number': 1},
        'env door y=180, substrate prior y=112 (68px mismatch)',
    ),
    (
        'door_at_30',
        {'door.position': np.array([30, 49, 49], dtype=int), 'door.number': 1},
        'env door y=30, substrate prior y=112 (82px mismatch)',
    ),
    (
        'door_at_200',
        {'door.position': np.array([200, 49, 49], dtype=int), 'door.number': 1},
        'env door y=200, substrate prior y=112 (88px mismatch)',
    ),
    (
        'door_at_200_narrow',
        {
            'door.position': np.array([200, 49, 49], dtype=int),
            'door.number': 1,
            'door.size': np.array([6, 14, 14], dtype=int),   # 6px slit; default=14
        },
        'env door y=200, size=6px, substrate prior y=112 — hardest: far + narrow',
    ),
    (
        'door_at_180_slow',
        {
            'door.position': np.array([180, 49, 49], dtype=int),
            'door.number': 1,
            'agent.speed': np.array([3.0], dtype=np.float32),
        },
        'env door y=180 + agent slow (speed=3): compound mismatch',
    ),
]


def run_regime(
    label: str,
    env_init: dict,
    description: str,
    num_eval: int = 50,
    seed: int = 42,
) -> dict:
    # Fixed prior: substrate always uses default dynamics (door at y=112).
    # This is what the paper claims generalises — we are stress-testing that claim.
    model = Pyo3SubstrateCostModel()

    world_kwargs: dict = {
        'num_envs': num_eval,
        'image_shape': (224, 224),
        'max_episode_steps': 100,
    }
    if env_init:
        world_kwargs['init_value'] = env_init

    world = swm.World('swm/TwoRoom-v1', **world_kwargs)
    solver = CEMSolver(model=model, num_samples=300, n_steps=30, topk=30, batch_size=25)
    config = PlanConfig(horizon=5, receding_horizon=5, action_block=5)
    policy = WorldModelPolicy(solver=solver, config=config)
    world.set_policy(policy)

    t0 = time.time()
    results = world.evaluate(episodes=num_eval, seed=seed)
    elapsed = time.time() - t0

    env_init_s = {
        k: v.tolist() if isinstance(v, np.ndarray) else v
        for k, v in env_init.items()
    }
    return {
        'label': label,
        'description': description,
        'env_init_value': env_init_s,
        'success_rate_pct': float(results['success_rate']),
        'episodes_solved': int(results['episode_successes'].sum()),
        'episodes_total': num_eval,
        'wall_clock_s': elapsed,
        'wall_clock_per_episode_ms': elapsed / num_eval * 1000.0,
    }


def main(
    num_eval: int = 50,
    seed: int = 42,
    out: str = 'results/ood_harder_substrate.json',
) -> dict:
    rows: list[dict] = []
    for label, env_init, description in HARDER_REGIMES:
        print(f'[HARDER] running {label}: {description}')
        try:
            row = run_regime(label, env_init, description,
                             num_eval=num_eval, seed=seed)
        except Exception as e:
            row = {'label': label, 'description': description, 'error': repr(e)}
        rows.append(row)
        sr = row.get('success_rate_pct', 'ERR')
        print(f'[HARDER]   → {label}: {sr}%')

    summary = {
        'label': 'ood_harder_substrate_fixed_prior',
        'env': 'swm/TwoRoom-v1',
        'model': 'Pyo3SubstrateCostModel (Rust v2, fixed default prior door_y=112)',
        'substrate_prior': {'door_positions': [112.0], 'door_half_extents': [14.0]},
        'config': {
            'num_eval': num_eval,
            'seed': seed,
            'cem': {'num_samples': 300, 'n_steps': 30, 'topk': 30},
            'plan': {'horizon': 5, 'receding_horizon': 5, 'action_block': 5},
        },
        'regimes': rows,
        'timestamp': datetime.utcnow().isoformat() + 'Z',
    }

    out_path = Path(__file__).resolve().parents[2] / out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2) + '\n')

    print('=' * 80)
    print(' HARDER OOD — substrate fixed_prior (door_y=112), env door varied')
    print('=' * 80)
    print(f'  {"Regime":<22} {"Success":>8}   Description')
    print('-' * 80)
    for r in rows:
        if 'error' in r:
            print(f'  {r["label"]:<22}    ERROR: {r["error"][:40]}')
        else:
            sr = r['success_rate_pct']
            ms = r['wall_clock_per_episode_ms']
            print(f'  {r["label"]:<22} {sr:>7.1f}%   {r["description"]}  ({ms:.0f}ms/ep)')
    print('=' * 80)
    print(f'saved to {out_path}')
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-eval', type=int, default=50)
    parser.add_argument('--seed',     type=int, default=42)
    parser.add_argument('--out',      type=str,
                        default='results/ood_harder_substrate.json')
    args = parser.parse_args()
    main(num_eval=args.num_eval, seed=args.seed, out=args.out)
