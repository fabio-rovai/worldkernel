"""Trick Room v0 — baselines on Two-Room.

Runs the trivial / built-in baselines:
  - RandomPolicy (action_space.sample())
  - ExpertPolicy (upstream weak expert with action_noise=2.0, repeat=5%)

These bracket the substrate result without requiring any trained model.
Neural baselines (GCBC / LeWM / DINO-WM / TD-MPC2) need separate training
on the tworoom_expert dataset and a GPU; deferred.
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
from stable_worldmodel.envs.two_room import ExpertPolicy
from stable_worldmodel.policy import RandomPolicy


def run(policy_name: str, policy, num_eval: int = 50, seed: int = 42) -> dict:
    world = swm.World(
        'swm/TwoRoom-v1',
        num_envs=num_eval,
        image_shape=(224, 224),
        max_episode_steps=100,
    )
    world.set_policy(policy)
    t0 = time.time()
    results = world.evaluate(episodes=num_eval, seed=seed)
    elapsed = time.time() - t0
    return {
        'label': policy_name,
        'success_rate_pct': float(results['success_rate']),
        'episodes_solved': int(results['episode_successes'].sum()),
        'episodes_total': num_eval,
        'wall_clock_total_s': elapsed,
        'wall_clock_per_episode_ms': elapsed / num_eval * 1000.0,
        'training_trajectories': 0,
    }


def main(num_eval: int = 50, seed: int = 42,
         out: str = 'results/baselines_two_room.json'):
    rows = []
    print('[baseline] running RandomPolicy...')
    rows.append(run('random_policy', RandomPolicy(), num_eval, seed))
    print('[baseline] running ExpertPolicy (action_noise=2.0, repeat=5%)...')
    rows.append(run('expert_policy_upstream_weak',
                    ExpertPolicy(action_noise=2.0, action_repeat_prob=0.05),
                    num_eval, seed))

    summary = {
        'env': 'swm/TwoRoom-v1',
        'config': {'num_eval': num_eval, 'seed': seed},
        'baselines': rows,
        'timestamp': datetime.utcnow().isoformat() + 'Z',
    }
    out_path = Path(__file__).resolve().parents[2] / out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2) + '\n')

    print('=' * 64)
    print(' TRICK ROOM BASELINES — Two-Room (no training)')
    print('=' * 64)
    print(f'{"policy":<32} {"success":>10} {"solved":>10} {"ms/ep":>10}')
    for r in rows:
        print(f'{r["label"]:<32} {r["success_rate_pct"]:>9.1f}% '
              f'{r["episodes_solved"]:>3}/{r["episodes_total"]:<6} '
              f'{r["wall_clock_per_episode_ms"]:>9.1f}')
    print('=' * 64)
    print(f'saved to {out_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-eval', type=int, default=50)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--out', type=str, default='results/baselines_two_room.json')
    args = parser.parse_args()
    main(num_eval=args.num_eval, seed=args.seed, out=args.out)
