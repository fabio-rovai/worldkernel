"""Trick Room v2 — OOD eval with Pyo3SubstrateCostModel (Rust, in-process).

Same regime table as ood_two_room.py (numpy v0).  Confirms that the Rust
physics (independently derived from tworoom.ttl) gives consistent OOD results.

Fixed_prior mode only (substrate keeps default dynamics regardless of regime) —
this is the hardest condition and the relevant one for the paper claim.
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


def run_regime(label: str, env_init: dict,
               num_eval: int = 50, seed: int = 42) -> dict:
    # Fixed prior: substrate always uses default dynamics
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

    env_init_s = {k: v.tolist() if isinstance(v, np.ndarray) else v
                  for k, v in env_init.items()}
    return {
        'label': label,
        'env_init_value': env_init_s,
        'success_rate_pct': float(results['success_rate']),
        'episodes_solved': int(results['episode_successes'].sum()),
        'episodes_total': num_eval,
        'wall_clock_s': elapsed,
        'wall_clock_per_episode_ms': elapsed / num_eval * 1000.0,
    }


def main(num_eval: int = 50, seed: int = 42,
         out: str = 'results/ood_two_room_pyo3.json') -> dict:
    rows: list[dict] = []
    for label, env_init in REGIMES:
        print(f'[OOD/pyo3] running {label}...')
        try:
            row = run_regime(label, env_init, num_eval=num_eval, seed=seed)
        except Exception as e:
            row = {'label': label, 'error': repr(e)}
        rows.append(row)
        print(f'[OOD/pyo3]   → {label}: {row.get("success_rate_pct", "ERR")}%')

    summary = {
        'label': 'pyo3_ood_fixed_prior',
        'env': 'swm/TwoRoom-v1',
        'model': 'Pyo3SubstrateCostModel (Rust v2, fixed default prior)',
        'config': {'num_eval': num_eval, 'seed': seed,
                   'cem': {'num_samples': 300, 'n_steps': 30, 'topk': 30},
                   'plan': {'horizon': 5, 'receding_horizon': 5, 'action_block': 5}},
        'regimes': rows,
        'timestamp': datetime.utcnow().isoformat() + 'Z',
    }

    out_path = Path(__file__).resolve().parents[2] / out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2) + '\n')

    print('=' * 68)
    print(' TRICK ROOM OOD — PyO3 substrate (fixed prior)')
    print('=' * 68)
    for r in rows:
        sr = r.get('success_rate_pct', 'ERR')
        ms = r.get('wall_clock_per_episode_ms', '?')
        print(f'  {r["label"]:<22} {sr}%   {ms:.0f}ms/ep' if isinstance(ms, float) else f'  {r["label"]}: ERROR')
    print('=' * 68)
    print(f'saved to {out_path}')
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-eval', type=int, default=50)
    parser.add_argument('--seed',     type=int, default=42)
    parser.add_argument('--out',      type=str, default='results/ood_two_room_pyo3.json')
    args = parser.parse_args()
    main(num_eval=args.num_eval, seed=args.seed, out=args.out)
