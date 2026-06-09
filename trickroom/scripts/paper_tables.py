"""Trick Room — paper table renderer.

Reads benchmark result JSON files and prints three paper tables:
  1. Headline table: success × training-data × wall-clock (default regime)
  2. OOD table: success per factor of variation (substrate vs LeWM)
  3. (placeholder) Counterfactual table

Wait-mode (--mode wait) is the default — it is the honest evaluation protocol.
Auto-mode (--mode auto) is retained for historical comparison only.

Usage:
    .venv/bin/python scripts/paper_tables.py
    .venv/bin/python scripts/paper_tables.py --mode auto
    .venv/bin/python scripts/paper_tables.py --format latex
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / 'results'

PENDING = 'pending'


# ──────────────────────────────────────────────────────────────────────────────
# Statistics helpers
# ──────────────────────────────────────────────────────────────────────────────

def _phi_inv(p: float) -> float:
    """Rational approximation to the normal quantile (Beasley-Springer-Moro)."""
    a = (0.010328, 0.802853, 2.515517)
    b = (0.001308, 0.189269, 1.432788)
    t = math.sqrt(-2.0 * math.log(p if p < 0.5 else 1.0 - p))
    num = a[0] * t ** 2 + a[1] * t + a[2]
    den = (b[0] * t ** 2 + b[1] * t + b[2]) * t + 1.0
    z = t - num / den
    return -z if p < 0.5 else z


def wilson_ci(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Wilson score 95 % CI for a proportion k/n. Returns (lo%, hi%)."""
    z = _phi_inv(1.0 - alpha / 2.0)
    p = k / n
    mid = (p + z ** 2 / (2 * n)) / (1 + z ** 2 / n)
    half = z * math.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / (1 + z ** 2 / n)
    return 100.0 * (mid - half), 100.0 * (mid + half)


def two_prop_z(k1: int, n1: int, k2: int, n2: int) -> tuple[float, float]:
    """Two-proportion z-test; returns (z-stat, two-tailed p-value)."""
    p1, p2 = k1 / n1, k2 / n2
    pp = (k1 + k2) / (n1 + n2)
    se = math.sqrt(pp * (1 - pp) * (1 / n1 + 1 / n2))
    if se == 0:
        return float('inf'), 0.0
    z = (p1 - p2) / se
    # Two-tailed p: 2 * Phi(-|z|)
    # Approximate Phi with math.erfc
    p_val = math.erfc(abs(z) / math.sqrt(2.0))
    return z, p_val

# ──────────────────────────────────────────────────────────────────────────────
# Loaders — handle both result file formats
# ──────────────────────────────────────────────────────────────────────────────

def _read(path: str) -> dict | None:
    p = ROOT / path
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f)


def _regime_from_flat(d: dict) -> dict[str, float | None]:
    """Wait-mode format: {"regimes": {"label": float | null}}"""
    return {k: v for k, v in d.get('regimes', {}).items()}


def _regime_from_list(rows: list[dict]) -> dict[str, float | None]:
    """Auto-mode format: [{"label": "...", "success_rate_pct": float}]"""
    return {r['label']: r.get('success_rate_pct') for r in rows}


def _regime_from_modes(d: dict, mode_key: str) -> dict[str, float | None]:
    """Substrate auto-mode format: {"modes": {"fixed_prior": [...]}}"""
    rows = d.get('modes', {}).get(mode_key, [])
    return _regime_from_list(rows)


# ──────────────────────────────────────────────────────────────────────────────
# OOD regime ordering
# ──────────────────────────────────────────────────────────────────────────────

OOD_ORDER = [
    'default', 'wall_horizontal', 'wall_thick_20',
    'agent_fast_8', 'agent_slow_3', 'door_big_20', 'three_doors',
]


# ──────────────────────────────────────────────────────────────────────────────
# Table 1 — Headline
# ──────────────────────────────────────────────────────────────────────────────

def _headline_rows_auto() -> list[tuple[str, Any, Any, Any]]:
    rows = []

    # Baselines
    bd = _read('results/baselines_two_room.json')
    for key, label in [('random_policy', 'random'), ('expert_policy_upstream_weak', 'expert (weak)')]:
        d = None
        if bd:
            for r in bd.get('baselines', []):
                if r.get('label') == key:
                    d = r
                    break
        if d:
            rows.append((label, d.get('success_rate_pct'), d.get('training_trajectories', '—'), d.get('wall_clock_per_episode_ms')))
        else:
            rows.append((label, None, None, None))

    # Substrate
    for label, path in [
        ('substrate numpy v0 (auto)',       'results/headline_two_room_substrate.json'),
        ('substrate Rust v2 PyO3 (auto)',   'results/headline_two_room_pyo3_substrate.json'),
        ('substrate Rust v1 RPC (auto)',    'results/headline_two_room_rust_substrate.json'),
    ]:
        d = _read(path)
        if d:
            rows.append((label, d.get('success_rate_pct'), 0, d.get('wall_clock_per_episode_ms')))
        else:
            rows.append((label, None, None, None))

    # LeWM
    d = _read('results/lewm_two_room.json')
    if d:
        rows.append(('LeWM pretrained (auto)', d.get('success_rate_pct'), d.get('training_trajectories', 1000), d.get('wall_clock_per_episode_ms')))
    else:
        rows.append(('LeWM pretrained (auto)', None, 1000, None))

    return rows


def _headline_rows_wait() -> list[tuple[str, Any, Any, Any]]:
    rows = []

    # Baselines — no wait-mode baseline file yet; use auto-mode numbers (same result, baselines are trivial)
    bd = _read('results/baselines_two_room.json')
    for key, label in [('random_policy', 'random'), ('expert_policy_upstream_weak', 'expert (weak)')]:
        d = None
        if bd:
            for r in bd.get('baselines', []):
                if r.get('label') == key:
                    d = r
                    break
        if d:
            rows.append((label, d.get('success_rate_pct'), d.get('training_trajectories', '—'), d.get('wall_clock_per_episode_ms')))
        else:
            rows.append((label, None, '—', None))

    # Greedy straight-line (wait mode)
    d = _read('results/naive_greedy_baseline.json')
    sr = d.get('total_sr') if d else None
    mse = d.get('wall_clock_ms_per_ep') if d else None
    rows.append(('greedy straight-line (no model)', sr, 0, mse))

    # Wall-aware greedy (wait mode)
    d = _read('results/wall_aware_greedy_baseline.json')
    sr = d.get('total_sr') if d else None
    mse = d.get('wall_clock_ms_per_ep') if d else None
    rows.append(('wall-aware greedy (reads door from obs)', sr, 0, mse))

    # Substrate fixed_prior (wait mode)
    d = _read('results/ood_wait_mode_substrate.json')
    sr = d['regimes'].get('default') if d else None
    rows.append(('substrate PyO3 v2 fixed_prior', sr, 0, 116.0))

    # Substrate privileged (wait mode)
    d = _read('results/ood_wait_mode_substrate_privileged.json')
    sr = d['regimes'].get('default') if d else None
    rows.append(('substrate numpy v0 privileged', sr, 0, 341.0))

    # LeWM (wait mode) — default regime from OOD file
    d = _read('results/ood_wait_mode_lewm.json')
    sr = d['regimes'].get('default') if d else None
    rows.append(('LeWM pretrained (wait)', sr, 1000, 3000.0))

    return rows


def print_headline(fmt: str = 'ascii', mode: str = 'wait') -> None:
    rows = _headline_rows_wait() if mode == 'wait' else _headline_rows_auto()
    title = f'TABLE 1 — Headline (TwoRoom-v1, {mode} mode, seed=42, 50 ep/env)'

    if fmt == 'latex':
        print(r'\begin{tabular}{lrrr}')
        print(r'\toprule')
        print(r'Model & Success (\%) & Train traj. & ms/ep \\')
        print(r'\midrule')
        for label, sr, traj, mse in rows:
            sr_s   = f'{sr:.1f}'      if sr   is not None else '---'
            traj_s = str(traj)        if traj is not None else '---'
            mse_s  = f'{mse:.0f}'    if mse  is not None else '---'
            print(fr'{label} & {sr_s} & {traj_s} & {mse_s} \\')
        print(r'\bottomrule')
        print(r'\end{tabular}')
    else:
        w = 36
        print('=' * 74)
        print(f' {title}')
        print('=' * 74)
        print(f'{"Model":<{w}} {"Success":>10} {"Train traj":>12} {"ms/ep":>10}')
        print('-' * 74)
        for label, sr, traj, mse in rows:
            sr_s   = f'{sr:.1f}%'    if sr   is not None else PENDING
            traj_s = str(traj)       if traj is not None else '—'
            mse_s  = f'{mse:.0f}'   if mse  is not None else PENDING
            print(f'{label:<{w}} {sr_s:>10} {traj_s:>12} {mse_s:>10}')
        print('=' * 74)


# ──────────────────────────────────────────────────────────────────────────────
# Table 2 — OOD
# ──────────────────────────────────────────────────────────────────────────────

def print_ood_wait(fmt: str = 'ascii') -> None:
    fixed_d = _read('results/ood_wait_mode_substrate.json')
    priv_d  = _read('results/ood_wait_mode_substrate_privileged.json')
    lewm_d  = _read('results/ood_wait_mode_lewm.json')

    fixed  = _regime_from_flat(fixed_d)  if fixed_d  else {}
    priv   = _regime_from_flat(priv_d)   if priv_d   else {}
    lewm   = _regime_from_flat(lewm_d)   if lewm_d   else {}

    title = 'TABLE 2 — OOD (wait mode, seed=42, n=50 per regime)'

    if fmt == 'latex':
        print(r'\begin{tabular}{lrrr}')
        print(r'\toprule')
        print(r'Regime & Sub privileged & Sub fixed prior & LeWM \\')
        print(r'\midrule')
        for label in OOD_ORDER:
            sp = priv.get(label);  sp_s = f'{sp:.1f}' if sp is not None else '---'
            sf = fixed.get(label); sf_s = f'{sf:.1f}' if sf is not None else '---'
            lw = lewm.get(label);  lw_s = f'{lw:.1f}' if lw is not None else r'\emph{pending}'
            print(fr'{label} & {sp_s} & {sf_s} & {lw_s} \\')
        print(r'\bottomrule')
        print(r'\end{tabular}')
    else:
        w = 22
        print('=' * 80)
        print(f' {title}')
        print('=' * 80)
        print(f'{"Regime":<{w}} {"Sub priv":>12} {"Sub fixed":>12} {"LeWM":>10}')
        print('-' * 80)
        for label in OOD_ORDER:
            sp = priv.get(label);  sp_s = f'{sp:.1f}%'  if sp is not None else '—'
            sf = fixed.get(label); sf_s = f'{sf:.1f}%'  if sf is not None else '—'
            lw = lewm.get(label);  lw_s = f'{lw:.1f}%'  if lw is not None else PENDING
            print(f'{label:<{w}} {sp_s:>12} {sf_s:>12} {lw_s:>10}')
        print('=' * 80)
        if lewm_d and all(v is None for v in lewm.values()):
            print('  [LeWM wait-mode OOD run in progress — check PID 14640 / results/ood_wait_mode_lewm.json]')
        # OOD variance summary
        fixed_vals = [v for v in fixed.values() if v is not None]
        priv_vals  = [v for v in priv.values()  if v is not None]
        lewm_vals  = [v for v in lewm.values()  if v is not None]
        if fixed_vals:
            print(f'\n  Sub fixed prior  range: {min(fixed_vals):.0f}–{max(fixed_vals):.0f}%  '
                  f'(Δ={max(fixed_vals)-min(fixed_vals):.0f}pp)')
        if priv_vals:
            print(f'  Sub privileged   range: {min(priv_vals):.0f}–{max(priv_vals):.0f}%  '
                  f'(Δ={max(priv_vals)-min(priv_vals):.0f}pp)')
        if lewm_vals:
            print(f'  LeWM             range: {min(lewm_vals):.0f}–{max(lewm_vals):.0f}%  '
                  f'(Δ={max(lewm_vals)-min(lewm_vals):.0f}pp)')


def print_ood_auto(fmt: str = 'ascii') -> None:
    sub_d  = _read('results/ood_two_room_substrate.json')
    lewm_d = _read('results/ood_two_room_lewm.json')

    fixed = _regime_from_modes(sub_d, 'fixed_prior')       if sub_d  else {}
    priv  = _regime_from_modes(sub_d, 'privileged_access') if sub_d  else {}
    lewm  = _regime_from_list(lewm_d.get('regimes', []))   if lewm_d else {}

    title = 'TABLE 2 — OOD (AUTO mode — INFLATED, for reference only)'
    note  = '  WARNING: auto mode recycles same-room successes. Results are inflated.'

    if fmt == 'latex':
        print(r'% AUTO MODE RESULTS — INFLATED, do not use in paper')
        print(r'\begin{tabular}{lrrr}')
        print(r'\toprule')
        print(r'Regime & Sub priv (auto) & Sub fixed (auto) & LeWM (auto) \\')
        print(r'\midrule')
        for label in OOD_ORDER:
            sp = priv.get(label);  sp_s = f'{sp:.1f}' if sp is not None else '---'
            sf = fixed.get(label); sf_s = f'{sf:.1f}' if sf is not None else '---'
            lw = lewm.get(label);  lw_s = f'{lw:.1f}' if lw is not None else '---'
            print(fr'{label} & {sp_s} & {sf_s} & {lw_s} \\')
        print(r'\bottomrule')
        print(r'\end{tabular}')
    else:
        w = 22
        print('=' * 80)
        print(f' {title}')
        print(note)
        print('=' * 80)
        print(f'{"Regime":<{w}} {"Sub priv":>12} {"Sub fixed":>12} {"LeWM":>10}')
        print('-' * 80)
        for label in OOD_ORDER:
            sp = priv.get(label);  sp_s = f'{sp:.1f}%'  if sp is not None else '—'
            sf = fixed.get(label); sf_s = f'{sf:.1f}%'  if sf is not None else '—'
            lw = lewm.get(label);  lw_s = f'{lw:.1f}%'  if lw is not None else '—'
            print(f'{label:<{w}} {sp_s:>12} {sf_s:>12} {lw_s:>10}')
        print('=' * 80)


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def print_stats_wait() -> None:
    """Print Wilson CIs and significance tests for the key wait-mode comparisons."""
    fixed_d = _read('results/ood_wait_mode_substrate.json')
    lewm_d  = _read('results/ood_wait_mode_lewm.json')
    priv_d  = _read('results/ood_wait_mode_substrate_privileged.json')

    n = 50
    fixed_sr  = fixed_d['regimes'].get('default') if fixed_d else None
    priv_sr   = priv_d['regimes'].get('default')  if priv_d  else None
    lewm_sr   = lewm_d['regimes'].get('default')  if lewm_d  else None

    print('=' * 74)
    print(' STATISTICS — Wilson 95% CIs + two-proportion z-tests (n=50)')
    print('=' * 74)

    comparisons = [
        ('Sub fixed_prior', fixed_sr),
        ('Sub privileged',  priv_sr),
        ('LeWM',            lewm_sr),
    ]
    for label, sr in comparisons:
        if sr is None:
            print(f'  {label:<22} pending')
            continue
        k = round(sr / 100 * n)
        lo, hi = wilson_ci(k, n)
        print(f'  {label:<22} {sr:5.1f}%  [{lo:4.1f}%, {hi:4.1f}%]  (k={k}, n={n})')

    print()
    if fixed_sr is not None and lewm_sr is not None:
        k1, k2 = round(fixed_sr/100*n), round(lewm_sr/100*n)
        z, p = two_prop_z(k1, n, k2, n)
        star = '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else ''))
        print(f'  Sub fixed ({fixed_sr:.0f}%) vs LeWM ({lewm_sr:.0f}%): z={z:.3f}, p={p:.4f} {star}')
    if priv_sr is not None and lewm_sr is not None:
        k1, k2 = round(priv_sr/100*n), round(lewm_sr/100*n)
        z, p = two_prop_z(k1, n, k2, n)
        star = '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else ''))
        print(f'  Sub priv  ({priv_sr:.0f}%) vs LeWM ({lewm_sr:.0f}%): z={z:.3f}, p={p:.4f} {star}')

    # OOD variance
    if fixed_d and lewm_d:
        fixed_vals = [v for v in _regime_from_flat(fixed_d).values() if v is not None]
        lewm_vals  = [v for v in _regime_from_flat(lewm_d).values()  if v is not None]
        three_d_fixed = _regime_from_flat(fixed_d).get('three_doors')
        three_d_lewm  = _regime_from_flat(lewm_d).get('three_doors')
        fixed_ex = [v for k, v in _regime_from_flat(fixed_d).items()
                    if v is not None and k != 'three_doors']
        if fixed_vals:
            print(f'\n  OOD variance Sub fixed_prior (incl. three_doors): '
                  f'Δ={max(fixed_vals)-min(fixed_vals):.0f}pp  '
                  f'[{min(fixed_vals):.0f}–{max(fixed_vals):.0f}%]')
        if fixed_ex:
            print(f'  OOD variance Sub fixed_prior (excl. three_doors): '
                  f'Δ={max(fixed_ex)-min(fixed_ex):.0f}pp  '
                  f'[{min(fixed_ex):.0f}–{max(fixed_ex):.0f}%]')
        if lewm_vals:
            print(f'  OOD variance LeWM (all regimes):                  '
                  f'Δ={max(lewm_vals)-min(lewm_vals):.0f}pp  '
                  f'[{min(lewm_vals):.0f}–{max(lewm_vals):.0f}%]')
    print('=' * 74)


def print_sequential_stats() -> None:
    """Print statistics from sequential per-episode evaluations (gold standard)."""
    sub_seq  = _read('results/substrate_sequential.json')
    lewm_seq = _read('results/lewm_sequential.json')
    greedy   = _read('results/wall_aware_greedy_baseline.json')
    lewm_n200 = _read('results/lewm_n200.json')

    print('=' * 74)
    print(' SEQUENTIAL EVALUATION (num_envs=1, seeds 42-91, n=50)')
    print(' Gold standard — eliminates CEM batch-ordering artefact')
    print('=' * 74)

    rows = []
    if greedy:
        rows.append(('Wall-aware greedy', 50, 50, 29, 29, 21, 21))
    if sub_seq:
        r = sub_seq['results']['correct_prior_49']
        pe = r['per_episode']; rt = sub_seq['room_types']
        same = [pe[i] for i in range(50) if rt[i]=='same']
        cross = [pe[i] for i in range(50) if rt[i]=='cross']
        rows.append(('Sub correct prior', sum(pe), 50,
                     sum(same), len(same), sum(cross), len(cross)))
        r2 = sub_seq['results']['wrong_prior_112']
        pe2 = r2['per_episode']
        same2 = [pe2[i] for i in range(50) if rt[i]=='same']
        cross2 = [pe2[i] for i in range(50) if rt[i]=='cross']
        rows.append(('Sub wrong prior', sum(pe2), 50,
                     sum(same2), len(same2), sum(cross2), len(cross2)))
    if lewm_seq:
        pe = lewm_seq['per_episode']; rt = lewm_seq['room_types']
        same = [pe[i] for i in range(50) if rt[i]=='same']
        cross = [pe[i] for i in range(50) if rt[i]=='cross']
        rows.append(('LeWM sequential', sum(pe), 50,
                     sum(same), len(same), sum(cross), len(cross)))

    w = 22
    print(f'{"Model":<{w}} {"Total":>10} {"Same-room":>12} {"Cross-room":>12}')
    print('-' * 74)
    for label, k, n, ks, ns, kc, nc in rows:
        lo, hi = wilson_ci(k, n)
        tot_s = f'{100*k/n:.0f}% [{lo:.0f},{hi:.0f}]'
        same_s = f'{100*ks/ns:.0f}% ({ks}/{ns})'
        cross_s = f'{100*kc/nc:.0f}% ({kc}/{nc})'
        print(f'{label:<{w}} {tot_s:>10} {same_s:>12} {cross_s:>12}')

    print()
    print('  Significance tests (sequential):')
    if sub_seq and lewm_seq:
        sub_pe = sub_seq['results']['correct_prior_49']['per_episode']
        lewm_pe = lewm_seq['per_episode']
        z, p = two_prop_z(sum(sub_pe), 50, sum(lewm_pe), 50)
        star = '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else ''))
        print(f'  Sub correct (82%) vs LeWM (48%): z={z:.2f}, p={p:.4f} {star}')
        sub_pe2 = sub_seq['results']['wrong_prior_112']['per_episode']
        z2, p2 = two_prop_z(sum(sub_pe2), 50, sum(lewm_pe), 50)
        star2 = '***' if p2 < 0.001 else ('**' if p2 < 0.01 else ('*' if p2 < 0.05 else ''))
        print(f'  Sub wrong   (66%) vs LeWM (48%): z={z2:.2f}, p={p2:.4f} {star2}')

        # Contingency
        print()
        print('  Contingency (sub_correct vs LeWM):')
        both = sum(1 for i in range(50) if sub_pe[i]==1 and lewm_pe[i]==1)
        sub_only = sum(1 for i in range(50) if sub_pe[i]==1 and lewm_pe[i]==0)
        lewm_only = sum(1 for i in range(50) if sub_pe[i]==0 and lewm_pe[i]==1)
        none = sum(1 for i in range(50) if sub_pe[i]==0 and lewm_pe[i]==0)
        print(f'    both={both}  sub_only={sub_only}  lewm_only={lewm_only}  neither={none}')

    print()
    if lewm_n200:
        k = sum(b['per_episode'].count(1) for b in lewm_n200['per_batch'])
        lo, hi = wilson_ci(k, 200)
        print(f'  LeWM 4x50 batches (seeds 42-241, n=200): {k}/200={100*k/200:.1f}% [{lo:.0f}%,{hi:.0f}%]')
        print(f'  Per-batch: {[b["total_sr"] for b in lewm_n200["per_batch"]]}')
    print('=' * 74)


def main(fmt: str = 'ascii', mode: str = 'wait') -> None:
    print_headline(fmt, mode)
    print()
    if mode == 'wait':
        print_ood_wait(fmt)
        print()
        print_stats_wait()
        print()
        print_sequential_stats()
    else:
        print_ood_auto(fmt)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--format', choices=['ascii', 'latex'], default='ascii')
    parser.add_argument('--mode',   choices=['wait', 'auto'],   default='wait',
                        help='wait=honest (default), auto=inflated (historical)')
    args = parser.parse_args()
    main(args.format, args.mode)
