"""The off-diagonal witness: the package's foundational claim, as a test."""

import numpy as np
import pytest

from worldkernel import TwoWorldKernel, frechet_pn_bounds, witness_pair


def test_witness_pair_identical_rungs_1_and_2():
    a, b = witness_pair()
    # rung 1: identical observational tables
    assert a.observational() == b.observational()
    # rung 2: identical interventional quantities
    assert a.diagonal == pytest.approx(b.diagonal)
    assert a.ace == pytest.approx(0.20)
    assert b.ace == pytest.approx(0.20)


def test_witness_pair_separates_at_rung_3():
    a, b = witness_pair()
    # the verified numbers: PN 0.286 (monotonic) vs 0.500 (independent)
    assert a.pn() == pytest.approx(2.0 / 7.0, abs=1e-9)
    assert b.pn() == pytest.approx(0.5, abs=1e-9)
    assert abs(a.pn() - b.pn()) > 0.2


def test_joint_is_a_distribution():
    for k in witness_pair():
        j = k.joint()
        assert all(v >= -1e-12 for v in j.values())
        assert sum(j.values()) == pytest.approx(1.0)


def test_kernel_is_admissible():
    for k in witness_pair():
        assert k.is_psd()
        assert k.admissible()


def test_pn_inside_frechet_interval():
    lo, hi = frechet_pn_bounds(0.5, 0.7)
    for k in witness_pair():
        assert lo - 1e-12 <= k.pn() <= hi + 1e-12
    # the interval is non-degenerate: rung-1/2 data do not pin PN down
    assert hi - lo > 0.2


def test_coupling_outside_frechet_box_rejected():
    with pytest.raises(ValueError):
        TwoWorldKernel(0.5, 0.7, p11=0.65)  # above min(r0, r1)
    with pytest.raises(ValueError):
        TwoWorldKernel(0.8, 0.9, p11=0.5)  # below r0 + r1 - 1


def test_pns_identity():
    a, _ = witness_pair()
    assert a.pns() == pytest.approx(a.pn() * a.r1)
    assert a.pns() == pytest.approx(a.ps() * (1.0 - a.r0))


def test_marginals_recoverable_from_joint():
    for k in witness_pair():
        j = k.joint()
        r0 = sum(v for (i, _), v in j.items() if i == 1)
        r1 = sum(v for (_, m), v in j.items() if m == 1)
        assert np.allclose([r0, r1], k.diagonal)
