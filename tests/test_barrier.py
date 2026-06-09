"""The Sly-Sun counting barrier: d_c and the finite-n correspondence."""

import random

import numpy as np
import pytest

from worldkernel import d_critical, order_parameter
from worldkernel.barrier import bp_marginals, exact_marginals, fixed_point_u, random_regular


def test_critical_degree_at_unit_fugacity():
    # the paper's constant: d_c = 5.1410415...
    assert d_critical(1.0) == pytest.approx(5.141, abs=2e-3)


def test_order_parameter_brackets_threshold():
    # the paper's values: (d-1)*eta = 0.98 at d=5, 1.11 at d=6
    assert order_parameter(5, 1.0) == pytest.approx(0.98, abs=0.01)
    assert order_parameter(6, 1.0) == pytest.approx(1.11, abs=0.01)
    assert order_parameter(5, 1.0) < 1.0 < order_parameter(6, 1.0)


def test_fixed_point_solves_equation():
    for d in (3, 5, 8):
        u = fixed_point_u(d, 1.0)
        assert u + u**d == pytest.approx(1.0, abs=1e-9)


def test_order_parameter_monotone_in_degree():
    ops = [order_parameter(d, 1.0) for d in range(2, 9)]
    assert all(a < b for a, b in zip(ops, ops[1:]))


def test_bp_error_rises_across_threshold():
    """Finite-n instrument check: BP tracks exact marginals at low degree and
    degrades above the threshold (monotonic correspondence, not a cliff)."""
    rng = random.Random(11)
    n = 12

    def mean_err(d: int, graphs: int = 3) -> float:
        errs = []
        for _ in range(graphs):
            adj = random_regular(n, d, rng)
            ex = exact_marginals(adj, n, 1.0)
            bp = bp_marginals(adj, n, 1.0)
            errs.append(float(np.mean(np.abs(ex - bp))))
        return float(np.mean(errs))

    below, above = mean_err(3), mean_err(7)
    assert below < 0.02  # accurate in the Weitz regime
    assert above > below  # and worse in the Sly-Sun regime
