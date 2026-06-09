"""Trick Room v0 — OOD eval across Two-Room factors of variation.

For each regime, pin one (or more) factors of the env's variation_space to a
non-default value, AND pass matching dynamics kwargs to SubstrateCostModel.
This is the "privileged access" condition: substrate has correct prior over
the regime's dynamics. Parametric baselines (trained on default regime) will
see the SAME pixel rendering and have to generalize without retraining.

Same CEM config as headline_two_room_substrate.py.
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
from stable_worldmodel.wm.substrate import SubstrateCostModel


REGIMES = [
    ('default', {}, {}),
    ('wall_horizontal',
        {'wall.axis': 0},
        {'wall_axis': 0}),
    ('wall_thick_20',
        {'wall.thickness': 20},
        {'wall_thickness': 20}),
    ('agent_fast_8',
        {'agent.speed': np.array([8.0], dtype=np.float32)},
        {'agent_speed': 8.0}),
    ('agent_slow_3',
        {'agent.speed': np.array([3.0], dtype=np.float32)},
        {'agent_speed': 3.0}),
    ('agent_big_10',
        {'agent.radius': np.array([10.0], dtype=np.float32)},
        {'agent_radius': 10.0}),
    ('door_big_20',
        {'door.size': np.array([20, 14, 14], dtype=int)},
        {'door_half_extents': (20, 14, 14)}),
    ('three_doors',
        {'door.number': 3,
         'door.position': np.array([50, 112, 175], dtype=int),
         'door.size': np.array([14, 14, 14], dtype=int)},
        {'num_doors': 3, 'door_half_extents': (14, 14, 14)}),
]


def run_regime(label: str, env_init: dict, substrate_kwargs: dict,
               num_eval: int = 50, seed: int = 42) -> dict:
    world_kwargs = {
        'env_name': 'swm/TwoRoom-v1',
        'num_envs': num_eval,
        'image_shape': (224, 224),
        'max_episode_steps': 100,
    }
    if env_init:
        world_kwargs['init_value'] = env_init

    world = swm.World(**world_kwargs)
    model = SubstrateCostModel(**substrate_kwargs)
    solver = CEMSolver(
        model=model, num_samples=300, n_steps=30, topk=30, batch_size=25,
    )
    config = PlanConfig(horizon=5, receding_horizon=5, action_block=5)
    policy = WorldModelPolicy(solver=solver, config=config)
    world.set_policy(policy)

    t0 = time.time()
    results = world.evaluate(episodes=num_eval, seed=seed)
    elapsed = time.time() - t0

    return {
        'label': label,
        'env_init_value': {k: v.tolist() if isinstance(v, np.ndarray) else v
                            for k, v in env_init.items()},
        'substrate_kwargs': {k: list(v) if isinstance(v, tuple) else v
                              for k, v in substrate_kwargs.items()},
        'success_rate_pct': float(results['success_rate']),
        'episodes_solved': int(results['episode_successes'].sum()),
        'episodes_total': num_eval,
        'wall_clock_s': elapsed,
        'wall_clock_per_episode_ms': elapsed / num_eval * 1000.0,
    }


def _run_pass(mode: str, num_eval: int, seed: int):
    rows = []
    for label, env_init, substrate_kwargs in REGIMES:
        # In fixed_prior mode, force substrate to keep defaults regardless of env regime.
        kwargs_for_substrate = {} if mode == 'fixed_prior' else substrate_kwargs
        print(f'[OOD/{mode}] running {label}...')
        try:
            row = run_regime(label, env_init, kwargs_for_substrate,
                             num_eval=num_eval, seed=seed)
        except Exception as e:
            row = {'label': label, 'error': repr(e),
                   'env_init_value': {k: v.tolist() if isinstance(v, np.ndarray) else v
                                       for k, v in env_init.items()},
                   'substrate_kwargs': {k: list(v) if isinstance(v, tuple) else v
                                         for k, v in kwargs_for_substrate.items()}}
        rows.append(row)
    return rows


def _print_table(mode: str, rows: list[dict]) -> None:
    print('=' * 64)
    print(f' TRICK ROOM OOD TABLE — {mode}')
    print('=' * 64)
    print(f'{"regime":<22} {"success":>10} {"solved":>10} {"ms/ep":>10}')
    for r in rows:
        if 'error' in r:
            print(f'{r["label"]:<22} ERROR: {r["error"][:60]}')
        else:
            print(f'{r["label"]:<22} {r["success_rate_pct"]:>9.1f}% '
                  f'{r["episodes_solved"]:>3}/{r["episodes_total"]:<6} '
                  f'{r["wall_clock_per_episode_ms"]:>9.1f}')
    print('=' * 64)


def main(num_eval: int = 50, seed: int = 42, out: str = 'results/ood_two_room_substrate.json'):
    priv_rows = _run_pass('privileged_access', num_eval, seed)
    fix_rows = _run_pass('fixed_prior', num_eval, seed)

    summary = {
        'env': 'swm/TwoRoom-v1',
        'config': {
            'num_eval': num_eval, 'seed': seed,
            'cem': {'num_samples': 300, 'n_steps': 30, 'topk': 30},
            'plan': {'horizon': 5, 'receding_horizon': 5, 'action_block': 5},
        },
        'modes': {
            'privileged_access': priv_rows,
            'fixed_prior': fix_rows,
        },
        'timestamp': datetime.utcnow().isoformat() + 'Z',
    }

    out_path = Path(__file__).resolve().parents[2] / out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2) + '\n')

    _print_table('privileged_access', priv_rows)
    _print_table('fixed_prior', fix_rows)
    print(f'saved to {out_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-eval', type=int, default=50)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--out', type=str, default='results/ood_two_room_substrate.json')
    args = parser.parse_args()
    main(num_eval=args.num_eval, seed=args.seed, out=args.out)
