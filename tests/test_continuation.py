"""The elastic wall: zero geometry governs analytic kernel access."""

import random

import numpy as np
import pytest

from worldkernel.barrier import random_regular
from worldkernel.continuation import (
    indep_poly,
    log_taylor_estimate,
    shearer_radius,
    zero_moat,
)
from worldkernel.tractable import hardcore_z


@pytest.fixture(scope="module")
def graphs():
    rng = random.Random(11)
    return {d: random_regular(16, d, rng) for d in (3, 6)}


def test_indep_poly_evaluates_to_z(graphs):
    for adj in graphs.values():
        c = indep_poly(adj)
        assert c[0] == 1.0  # the empty world
        assert float(c.sum()) == pytest.approx(hardcore_z(adj, 1.0), rel=1e-9)


def test_the_disk_always_fails(graphs):
    """Some zero always lies inside |t| < 1 (the product of root moduli is
    1/i_alpha << 1), so the Barvinok disk centred at 0 never reaches
    lam = 1: the elastic PATH framing is forced."""
    for adj in graphs.values():
        m = zero_moat(indep_poly(adj))
        assert m["min_root_modulus"] < 1.0


def test_truncation_obeys_the_moat(graphs):
    """log-Taylor truncation converges inside the closest zero and is
    badly wrong at t = 1, which lies outside the disk on every instance."""
    for adj in graphs.values():
        c = indep_poly(adj)
        moat = zero_moat(c)["min_root_modulus"]
        t_in = 0.5 * moat
        exact_in = float(sum(ck * t_in**k for k, ck in enumerate(c)))
        est = log_taylor_estimate(c, t_in, m=40)
        assert est == pytest.approx(exact_in, rel=1e-6)  # geometric convergence
        exact_1 = float(c.sum())
        est_1 = log_taylor_estimate(c, 1.0, m=40)
        assert abs(est_1 - exact_1) / exact_1 > 0.5  # divergence past the moat


def test_truncation_error_decays_geometrically_inside(graphs):
    c = indep_poly(graphs[3])
    moat = zero_moat(c)["min_root_modulus"]
    t = 0.6 * moat
    exact = float(sum(ck * t**k for k, ck in enumerate(c)))
    errs = [abs(log_taylor_estimate(c, t, m) - exact) / exact for m in (5, 10, 20)]
    assert errs[0] > errs[1] > errs[2]
    assert errs[2] < 1e-4


def test_segment_clearance_is_the_shearer_moat(graphs):
    """At enumerable sizes the clearance of the real segment [0, 1] is set
    by the negative-axis zeros near -lam*(d): above the worst-case Shearer
    radius, decreasing with degree, and with NO positive-real-part zeros at
    all (the honest negative finding: the asymptotic positive-axis pinching
    for d >= 6 is invisible at n = 16)."""
    moats = {}
    for d, adj in graphs.items():
        c = indep_poly(adj)
        m = zero_moat(c)
        moats[d] = m["segment_clearance"]
        assert m["segment_clearance"] >= shearer_radius(d)
        # clearance equals min modulus: the binding zeros sit near the
        # negative axis close to 0, none in the right half plane
        assert m["segment_clearance"] == pytest.approx(
            m["min_root_modulus"], rel=1e-9
        )
    assert moats[6] < moats[3]  # the moat narrows with degree


def test_shearer_radius_values():
    assert shearer_radius(3) == pytest.approx(4 / 27)
    assert shearer_radius(6) == pytest.approx(3125 / 46656)