"""Trick Room v0 — substrate cost model on upstream Two-Room MPC eval.

Matches the upstream `scripts/plan/config/tworoom.yaml` CEM config exactly:
horizon=5, receding_horizon=5, action_block=5, num_samples=300, n_steps=30,
topk=30, num_eval=50, seed=42.

Reports success rate, wall clock, and per-episode latency. Saves a JSON
result file under results/.

Usage:
    .venv/bin/python scripts/benchmark/trick_room_two_room.py
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
from stable_worldmodel.wm.substrate import SubstrateCostModel


def main(num_eval: int = 50, seed: int = 42, label: str = 'v0_python_substrate') -> dict:
    world = swm.World(
        'swm/TwoRoom-v1',
        num_envs=num_eval,
        image_shape=(224, 224),
        max_episode_steps=100,
    )
    model = SubstrateCostModel()
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

    t0 = time.time()
    results = world.evaluate(episodes=num_eval, seed=seed)
    elapsed = time.time() - t0

    summary = {
        'label': label,
        'env': 'swm/TwoRoom-v1',
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
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-eval', type=int, default=50)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--label', type=str, default='v0_python_substrate')
    parser.add_argument('--out', type=str, default='results/headline_two_room_substrate.json')
    args = parser.parse_args()

    summary = main(num_eval=args.num_eval, seed=args.seed, label=args.label)

    out_path = Path(__file__).resolve().parents[2] / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2) + '\n')

    print('=' * 60)
    print(f' TRICK ROOM HEADLINE — {summary["label"]}')
    print('=' * 60)
    print(f'  env                     {summary["env"]}')
    print(f'  success_rate            {summary["success_rate_pct"]:.1f}%')
    print(f'  episodes                {summary["episodes_solved"]}/{summary["episodes_total"]}')
    print(f'  wall_clock_total        {summary["wall_clock_total_s"]:.2f}s')
    print(f'  wall_clock_per_episode  {summary["wall_clock_per_episode_ms"]:.1f}ms')
    print(f'  training_trajectories   {summary["training_trajectories"]}')
    print(f'  saved to                {out_path}')
