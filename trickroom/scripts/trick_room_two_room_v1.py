"""Trick Room v1 — Rust substrate subprocess on Two-Room.

Same CEM config as v0 and upstream tworoom.yaml.  Difference: the world model
is now the compiled Rust binary (tworoom_substrate_rpc) bridged via
subprocess JSON-RPC, NOT the numpy port of env.py.

This is the claim: a typed-rule substrate derived from data/tbox/tworoom.ttl,
running in Rust as a separate process, zero training, same harness.

Usage:
    .venv/bin/python scripts/benchmark/trick_room_two_room_v1.py
"""

from __future__ import annotations

import argparse
import json
import time
import warnings
from datetime import datetime
from pathlib import Path

warnings.filterwarnings('ignore')

import stable_worldmodel as swm
from stable_worldmodel.policy import PlanConfig, WorldModelPolicy
from stable_worldmodel.solver import CEMSolver
from stable_worldmodel.wm.substrate import RustSubstrateCostModel


def main(num_eval: int = 50, seed: int = 42,
         out: str = 'results/headline_two_room_rust_substrate.json') -> dict:

    model = RustSubstrateCostModel()
    print(f'[v1] Rust substrate process spawned: {model._binary_path}')

    world = swm.World(
        'swm/TwoRoom-v1',
        num_envs=num_eval,
        image_shape=(224, 224),
        max_episode_steps=100,
    )
    solver = CEMSolver(
        model=model,
        num_samples=300,
        n_steps=30,
        topk=30,
        batch_size=25,
    )
    config = PlanConfig(horizon=5, receding_horizon=5, action_block=5)
    policy = WorldModelPolicy(solver=solver, config=config)
    world.set_policy(policy)

    print(f'[v1] running {num_eval} episodes (seed={seed})...')
    t0 = time.time()
    results = world.evaluate(episodes=num_eval, seed=seed)
    elapsed = time.time() - t0
    model.close()

    summary = {
        'label': 'v1_rust_substrate',
        'env': 'swm/TwoRoom-v1',
        'substrate_binary': str(model._binary_path),
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
        'training_trajectories': 0,
        'timestamp': datetime.utcnow().isoformat() + 'Z',
    }

    out_path = Path(__file__).resolve().parents[2] / out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2) + '\n')

    print('=' * 64)
    print(' TRICK ROOM v1 — Rust Substrate (Two-Room)')
    print('=' * 64)
    print(f'success_rate:  {summary["success_rate_pct"]:.1f}%')
    print(f'episodes:      {summary["episodes_solved"]}/{summary["episodes_total"]}')
    print(f'ms/episode:    {summary["wall_clock_per_episode_ms"]:.1f}')
    print(f'training:      {summary["training_trajectories"]} trajectories')
    print('=' * 64)
    print(f'saved to {out_path}')

    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-eval', type=int, default=50)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--out', type=str,
                        default='results/headline_two_room_rust_substrate.json')
    args = parser.parse_args()
    main(num_eval=args.num_eval, seed=args.seed, out=args.out)
