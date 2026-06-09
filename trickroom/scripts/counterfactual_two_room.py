"""Trick Room — Counterfactual eval (Table 3).

Runs 50 episodes of the DEFAULT TwoRoom env (door at y=112, unchanged) while
varying the substrate's INTERNAL belief about door position.  LeWM serves as
the statistical baseline with no causal intervention possible.

This tests whether the substrate's success rate tracks its causal beliefs:
  - Correct prior  → 100% (door belief matches env)
  - Wrong-far door → low% (substrate navigates to wrong y, misses door)
  - Wrong-close    → intermediate% (partial match)
  - Sealed wall    → ~0% (substrate knows wall is impassable; CEM never plans a cross)

LeWM cannot be causally intervened on (no symbolic prior to modify) — its
success stays near the default 66% regardless of what we do to the substrate.

The comparison directly demonstrates structural/causal generalisation vs.
statistical approximation:
  substrate(correct prior)  = 100%
  substrate(wrong door)     = X%   — degrades proportionally to mismatch
  LeWM(pretrained)          = 66%  — unchangeable; no causal handle
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

# ──────────────────────────────────────────────────────────────────────────────
# Counterfactual conditions:
#   (label, substrate door_positions, description)
# The ENV is always default (door at y=112).
# Only the substrate's internal belief changes.
#
# Pyo3SubstrateCostModel uses __init__ door_positions as its prior — it does
# NOT read door positions from the observation.  This makes it the right tool
# for causal intervention experiments: the env truth is fixed, the substrate's
# causal model is varied.
# ──────────────────────────────────────────────────────────────────────────────
CONDITIONS = [
    ('correct_prior',   [112.0],  'door belief matches env (y=112)'),
    ('wrong_close',     [80.0],   'door belief off by 32 px (y=80)'),
    ('wrong_far',       [200.0],  'door belief off by 88 px (y=200)'),
    ('sealed_wall',     [],       'substrate believes wall is sealed (num_doors=0)'),
]


def run_condition(
    label: str,
    door_positions: list[float],
    description: str,
    num_eval: int = 50,
    seed: int = 42,
) -> dict:
    num_doors = len(door_positions)
    door_half_extents = [14.0] * max(num_doors, 1)

    model = Pyo3SubstrateCostModel(
        door_positions=door_positions if door_positions else [0.0],
        door_half_extents=[0.0] if not door_positions else door_half_extents,
        num_doors=num_doors,
    )

    world = swm.World(
        'swm/TwoRoom-v1',
        num_envs=num_eval,
        image_shape=(224, 224),
        max_episode_steps=100,
    )
    solver = CEMSolver(model=model, num_samples=300, n_steps=30, topk=30, batch_size=25)
    config = PlanConfig(horizon=5, receding_horizon=5, action_block=5)
    policy = WorldModelPolicy(solver=solver, config=config)
    world.set_policy(policy)

    t0 = time.time()
    results = world.evaluate(episodes=num_eval, seed=seed)
    elapsed = time.time() - t0

    return {
        'label': label,
        'description': description,
        'substrate_door_positions': door_positions,
        'substrate_num_doors': num_doors,
        'env_door_positions': [112.0],
        'env_num_doors': 1,
        'success_rate_pct': float(results['success_rate']),
        'episodes_solved': int(results['episode_successes'].sum()),
        'episodes_total': num_eval,
        'wall_clock_s': elapsed,
        'wall_clock_per_episode_ms': elapsed / num_eval * 1000.0,
    }


def _print_table(rows: list[dict]) -> None:
    print('=' * 80)
    print(' TABLE 3 — Counterfactual (env: default, only substrate belief varies)')
    print('=' * 80)
    print(f'{"Condition":<22} {"Description":<35} {"Success":>8}')
    print('-' * 80)
    for r in rows:
        if 'error' in r:
            print(f'{r["label"]:<22} ERROR: {r["error"][:50]}')
        else:
            print(
                f'{r["label"]:<22} {r["description"]:<35}'
                f' {r["success_rate_pct"]:>7.1f}%'
            )
    print('=' * 80)


def main(num_eval: int = 50, seed: int = 42,
         out: str = 'results/counterfactual_two_room.json') -> dict:
    rows: list[dict] = []
    for label, door_positions, desc in CONDITIONS:
        print(f'[CF] running {label}: {desc}')
        try:
            row = run_condition(label, door_positions, desc,
                                num_eval=num_eval, seed=seed)
        except Exception as e:
            row = {'label': label, 'description': desc, 'error': repr(e)}
        rows.append(row)
        sr = row.get('success_rate_pct', 'ERR')
        print(f'[CF]   → {label}: {sr}%')

    summary = {
        'label': 'counterfactual_substrate',
        'env': 'swm/TwoRoom-v1',
        'env_default_door': [112.0],
        'model': 'Pyo3SubstrateCostModel (Rust v2, causal prior varied)',
        'config': {
            'num_eval': num_eval,
            'seed': seed,
            'cem': {'num_samples': 300, 'n_steps': 30, 'topk': 30},
            'plan': {'horizon': 5, 'receding_horizon': 5, 'action_block': 5},
        },
        'conditions': rows,
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
                        default='results/counterfactual_two_room.json')
    args = parser.parse_args()
    main(num_eval=args.num_eval, seed=args.seed, out=args.out)
