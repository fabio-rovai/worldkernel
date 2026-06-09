"""Mediation: the off-diagonal witness scaled to a nested counterfactual."""

import numpy as np
import pytest

from worldkernel import atom_count, nde_interval, random_reference, rung12_summary
from worldkernel.mediation import ATOMS, rung12_constraints


def test_atom_space():
    assert len(ATOMS) == 64
    assert atom_count(1) == 64
    assert atom_count(2) == 4096
    assert atom_count(3) == 4_194_304  # the counting barrier, concretely


def test_seeded_interval_spans_zero():
    p0 = random_reference(seed=0)
    lo, hi, _, _ = nde_interval(p0)
    # the verified instance: interval approximately [-0.381, +0.187]
    assert lo == pytest.approx(-0.381, abs=0.02)
    assert hi == pytest.approx(0.187, abs=0.02)
    assert lo < 0.0 < hi  # the sign of the NDE is genuinely undetermined
    assert (hi - lo) == pytest.approx(0.568, abs=0.03)


def test_endpoint_models_share_rungs_1_and_2():
    p0 = random_reference(seed=0)
    A, b = rung12_constraints(p0)
    _, _, p_lo, p_hi = nde_interval(p0)
    assert np.allclose(A @ p_lo, b, atol=1e-6)
    assert np.allclose(A @ p_hi, b, atol=1e-6)
    # and they are genuinely different distributions
    assert not np.allclose(p_lo, p_hi, atol=1e-3)


def test_point_identified_when_no_mediator_freedom():
    # degenerate reference: a single atom has all the mass, so rungs 1-2
    # pin the law and the interval collapses to a point
    p0 = np.zeros(len(ATOMS))
    p0[0] = 1.0
    lo, hi, _, _ = nde_interval(p0)
    assert hi - lo == pytest.approx(0.0, abs=1e-8)


def test_rung12_summary_keys():
    s = rung12_summary(random_reference(seed=0))
    assert len(s) == 8
    assert all(0.0 <= v <= 1.0 for v in s.values())
