"""The World Model Arena as an AISI Inspect eval: LLMs as contenders.

Packages the arena's identification questions as an Inspect AI task
(https://inspect.aisi.org.uk, the UK AI Security Institute's evaluation
framework, the same one ControlArena is built on). The model under
evaluation receives exactly what an experiment reveals (rung-1/2 data) and
is scored with the Winkler interval score against the hidden truth, with
the kernel's identified interval logged beside every sample as the
reference answer.

The model is told it may answer with an interval if the quantity is not
identified. That is the test: does an LLM world-model know what it cannot
know? A model that commits to a point on every unidentified query scores as
the arena's "independence predictor" does: 0% coverage, full penalty when
wrong. A model that returns valid sharp intervals matches the kernel.

Run (any Inspect-supported model):
  pip install inspect-ai worldkernel
  inspect eval integrations/inspect_arena.py --model anthropic/claude-sonnet-4-6
Smoke test without a model API:
  inspect eval integrations/inspect_arena.py --model mockllm/model --limit 4

Metrics: mean Winkler at alpha 0.2 and 0.02 (lower is better), coverage,
overclaim rate, and regret vs the kernel's interval on the same question.
"""

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
from inspect_ai import Task, task
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.scorer import Metric, Score, Target, metric, scorer
from inspect_ai.solver import generate, system_message

from worldkernel.arena import winkler

SYSTEM = (
    "You are evaluated as a world model. Each question gives you everything "
    "a randomized experiment can measure (observational and interventional "
    "quantities) and asks for a counterfactual quantity. Some of these are "
    "NOT identified by the data given: in that case the honest answer is an "
    "interval. Reply with either a single number, or an interval in the "
    "exact form [lo, hi]. No other text."
)


def _build_samples(seed: int, n_two_arm: int, n_mediation: int) -> list[Sample]:
    """Build LLM-facing samples directly (question text needs the raw
    rung-1/2 numbers, so we regenerate worlds here with the same seed
    discipline as the arena)."""
    rng = np.random.default_rng(seed)
    samples: list[Sample] = []

    from worldkernel import frechet_pn_bounds
    from worldkernel.mediation import (
        ATOMS,
        m_val,
        nde_interval_from_record,
        nde_vector,
        y_val,
    )

    for i in range(n_two_arm):
        r0, r1 = rng.uniform(0.15, 0.85, size=2)
        lo_box, hi_box = max(0.0, r0 + r1 - 1.0), min(r0, r1)
        p11 = rng.uniform(lo_box, hi_box)
        truth = (r1 - p11) / r1
        k_lo, k_hi = frechet_pn_bounds(r0, r1)
        q = (
            f"A randomized trial of a binary treatment X on outcome Y reports "
            f"P(Y=1|do(X=0)) = {r0:.3f} and P(Y=1|do(X=1)) = {r1:.3f}. "
            f"Among treated units with Y=1, what is the probability they "
            f"would have had Y=0 without treatment (probability of necessity)?"
        )
        samples.append(
            Sample(
                input=q,
                target=f"{truth:.6f}",
                id=f"two_arm_pn_{i}",
                metadata={
                    "world_class": "two_arm",
                    "truth": float(truth),
                    "kernel_lo": float(k_lo),
                    "kernel_hi": float(k_hi),
                },
            )
        )

    nde = nde_vector()
    for i in range(n_mediation):
        p0 = rng.dirichlet(np.ones(len(ATOMS)) * 0.3)
        truth = float(nde @ p0)

        def val(fn):
            return float(np.array([fn(iM, iY) for (iM, iY) in ATOMS]) @ p0)

        p_m = tuple(val(lambda iM, iY, x=x: m_val(iM, x) == 1) for x in (0, 1))
        p_my = {
            (x, m): val(
                lambda iM, iY, x=x, m=m: m_val(iM, x) == m
                and y_val(iY, x, m_val(iM, x)) == 1
            )
            for x in (0, 1)
            for m in (0, 1)
        }
        p_ydo = {
            (x, m): val(lambda iM, iY, x=x, m=m: y_val(iY, x, m) == 1)
            for x in (0, 1)
            for m in (0, 1)
        }
        k_lo, k_hi = nde_interval_from_record(p_m, p_my, p_ydo)
        q = (
            "Randomized mediation experiment, all binary, X -> M -> Y.\n"
            f"P(M=1|do X=0)={p_m[0]:.3f}, P(M=1|do X=1)={p_m[1]:.3f}\n"
            + ", ".join(
                f"P(M={m},Y=1|do X={x})={v:.3f}" for (x, m), v in p_my.items()
            )
            + "\n"
            + ", ".join(
                f"P(Y=1|do X={x},do M={m})={v:.3f}" for (x, m), v in p_ydo.items()
            )
            + "\nWhat is the Natural Direct Effect "
            "NDE = P(Y_{X=1,M=M_0}=1) - P(Y_{X=0,M=M_0}=1)?"
        )
        samples.append(
            Sample(
                input=q,
                target=f"{truth:.6f}",
                id=f"mediation_nde_{i}",
                metadata={
                    "world_class": "mediation",
                    "truth": truth,
                    "kernel_lo": float(k_lo),
                    "kernel_hi": float(k_hi),
                },
            )
        )
    return samples


def _parse_answer(text: str) -> tuple[float, float] | None:
    text = text.strip()
    m = re.search(r"\[\s*(-?\d*\.?\d+)\s*,\s*(-?\d*\.?\d+)\s*\]", text)
    if m:
        lo, hi = float(m.group(1)), float(m.group(2))
        return (lo, hi) if lo <= hi else (hi, lo)
    nums = re.findall(r"-?\d*\.?\d+", text)
    if nums:
        x = float(nums[-1])
        return (x, x)
    return None


def _mean_of(key: str) -> Metric:
    def compute(scores) -> float:
        vals = []
        for s in scores:
            sc = getattr(s, "score", s)  # SampleScore in newer Inspect, Score in older
            if sc.metadata and key in sc.metadata:
                vals.append(sc.metadata[key])
        return float(np.mean(vals)) if vals else float("nan")

    return compute


@metric
def winkler_02() -> Metric:
    return _mean_of("winkler_02")


@metric
def winkler_002() -> Metric:
    return _mean_of("winkler_002")


@metric
def coverage() -> Metric:
    return _mean_of("covered")


@metric
def overclaim() -> Metric:
    return _mean_of("overclaimed")


@metric
def regret_vs_kernel() -> Metric:
    return _mean_of("regret_vs_kernel")


@scorer(metrics=[winkler_02(), winkler_002(), coverage(), overclaim(), regret_vs_kernel()])
def winkler_scorer():
    async def score(state, target: Target) -> Score:
        md = state.metadata
        truth = float(md["truth"])
        parsed = _parse_answer(state.output.completion or "")
        if parsed is None:  # unparseable: worst case on the kernel's scale
            lo, hi = md["kernel_lo"], md["kernel_hi"]
            w02 = winkler(lo, hi, truth, 0.2) + 10.0
            return Score(
                value=w02,
                answer=state.output.completion,
                explanation="unparseable answer; penalized",
                metadata={
                    "winkler_02": w02,
                    "winkler_002": winkler(lo, hi, truth, 0.02) + 100.0,
                    "covered": 0.0,
                    "overclaimed": 1.0,
                    "regret_vs_kernel": 10.0,
                },
            )
        lo, hi = parsed
        w02 = winkler(lo, hi, truth, 0.2)
        w002 = winkler(lo, hi, truth, 0.02)
        kernel_w02 = winkler(md["kernel_lo"], md["kernel_hi"], truth, 0.2)
        covered = float(lo - 1e-9 <= truth <= hi + 1e-9)
        overclaimed = float((hi - lo) < 1e-12 and abs(lo - truth) > 0.02)
        return Score(
            value=w02,
            answer=f"[{lo:.4f}, {hi:.4f}]" if hi > lo else f"{lo:.4f}",
            explanation=f"truth {truth:.4f}; kernel [{md['kernel_lo']:.4f}, "
            f"{md['kernel_hi']:.4f}]",
            metadata={
                "winkler_02": w02,
                "winkler_002": w002,
                "covered": covered,
                "overclaimed": overclaimed,
                "regret_vs_kernel": w02 - kernel_w02,
            },
        )

    return score


@task
def world_model_arena(seed: int = 11, n_two_arm: int = 20, n_mediation: int = 10) -> Task:
    """LLM-as-world-model: identification questions, proper interval scoring."""
    return Task(
        dataset=MemoryDataset(_build_samples(seed, n_two_arm, n_mediation)),
        solver=[system_message(SYSTEM), generate()],
        scorer=winkler_scorer(),
    )
