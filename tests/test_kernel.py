"""The nested bounds: Frechet contains PSD contains exact."""

import numpy as np
import pytest

from worldkernel import CouplingKernel, exact_interval, frechet_interval

cvxpy = pytest.importorskip("cvxpy", reason="psd_interval requires cvxpy")
from worldkernel import psd_interval  # noqa: E402


RNG = np.random.default_rng(11)


def test_kernel_validation():
    with pytest.raises(ValueError):
        CouplingKernel([[1, 2, 3], [4, 5, 6]])  # not square
    with pytest.raises(ValueError):
        CouplingKernel([[0.5, 0.1], [0.4, 0.5]])  # not symmetric


def test_psd_and_admissible():
    good = CouplingKernel([[0.5, 0.35], [0.35, 0.7]])
    assert good.is_psd() and good.admissible()
    # symmetric, inside Frechet boxes, but not PSD
    bad = CouplingKernel([[0.04, 0.04, 0.0], [0.04, 0.04, 0.04], [0.0, 0.04, 0.04]])
    assert not bad.is_psd()
    assert not bad.admissible()


def test_nested_bounds_frechet_psd_exact():
    """The paper's verified ordering on 10 random instances at k=6."""
    for _ in range(10):
        d = RNG.uniform(0.15, 0.85, size=6)
        fl, fh = frechet_interval(d)
        pl, ph = psd_interval(d)
        el, eh = exact_interval(d)
        # PSD is a valid outer bound on the exact identified set
        assert pl <= el + 1e-5
        assert ph >= eh - 1e-5
        # and lives inside the Frechet box
        assert pl >= fl - 1e-5
        assert ph <= fh + 1e-5


def test_psd_strictly_tightens_frechet_at_scale():
    """The PSD constraint starts binding around k=16 and the relative gap
    grows with k (reaching ~8% by k=40). Test at k=20 where it is robust,
    a regime the 2^k exact LP can still reach but only barely (2^20 vars)."""
    tightened = 0
    for _ in range(5):
        d = RNG.uniform(0.15, 0.85, size=20)
        fl, fh = frechet_interval(d)
        pl, ph = psd_interval(d)
        if ((fh - fl) - (ph - pl)) / (fh - fl) > 1e-3:
            tightened += 1
    assert tightened >= 3


def test_pairwise_coherence_matches_offdiagonal_sum():
    M = np.array([[0.5, 0.3, 0.2], [0.3, 0.6, 0.25], [0.2, 0.25, 0.4]])
    assert CouplingKernel(M).pairwise_coherence() == pytest.approx(0.3 + 0.2 + 0.25)
