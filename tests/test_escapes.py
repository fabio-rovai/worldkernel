"""The three escape hatches: backdoor collapse, phase quotient, proofs."""

import numpy as np
import pytest

from worldkernel import (
    PhaseQuotient,
    SumcheckProver,
    backdoor_marginal,
    backdoor_z,
    find_backdoor,
    hub_world,
    kmm_exact_marginal,
    kmm_quotient,
    verify_backdoor,
    verify_z,
)
from worldkernel.barrier import exact_marginals
from worldkernel.tractable import hardcore_z


# ---- backdoor collapse ---------------------------------------------------------

def _enum_z(adj) -> float:
    n = len(adj)
    mask = [0] * n
    for i in range(n):
        for j in adj[i]:
            mask[i] |= 1 << j
    z = 0
    for s in range(1 << n):
        ok = True
        t = s
        while t:
            i = (t & -t).bit_length() - 1
            if mask[i] & s:
                ok = False
                break
            t &= t - 1
        z += ok
    return float(z)


@pytest.fixture(scope="module")
def hub():
    # 14-vertex 3-regular bulk + 2 hubs of degree 9: global degree 9 >> 6
    return hub_world(n_base=14, d_base=3, n_hubs=2, hub_degree=9)


def test_hub_world_breaks_the_degree_diagnosis(hub):
    assert max(len(a) for a in hub) >= 6  # 'hard' by the degree criterion


def test_greedy_backdoor_is_small_and_certified(hub):
    B = find_backdoor(hub)
    assert verify_backdoor(hub, B)
    assert len(B) <= 3  # the two hubs (plus at most one bulk vertex)
    assert not verify_backdoor(hub, set())  # without B the certificate fails


def test_backdoor_z_matches_enumeration(hub):
    B = find_backdoor(hub)
    assert backdoor_z(hub, B) == pytest.approx(_enum_z(hub), rel=1e-9)


def test_backdoor_marginal_matches_enumeration(hub):
    B = find_backdoor(hub)
    ex = exact_marginals(hub, len(hub), 1.0)
    for v in (0, 7, len(hub) - 1):  # bulk, bulk, hub
        assert backdoor_marginal(hub, v, B) == pytest.approx(ex[v], abs=1e-9)


def test_backdoor_survives_restriction(hub):
    """Conditioning deletes vertices; the same B stays a certificate."""
    B = find_backdoor(hub)
    restricted = [set(w for w in a if w != 0) for a in hub]
    restricted[0] = set()
    assert verify_backdoor(restricted, B)


# ---- phase quotient --------------------------------------------------------------

def test_kmm_quotient_matches_enumeration_small():
    m = 7  # n = 14: enumerable
    adj = [set(range(m, 2 * m)) for _ in range(m)] + [
        set(range(m)) for _ in range(m)
    ]
    ex = exact_marginals(adj, 2 * m, 1.0)
    pq, _ = kmm_quotient(m)
    lo, hi = pq.interval()
    assert lo == pytest.approx(hi, abs=1e-12)  # weights identified: a point
    assert lo == pytest.approx(ex[0], abs=1e-9)
    assert kmm_exact_marginal(m) == pytest.approx(ex[0], abs=1e-9)


def test_kmm_quotient_at_astronomical_scale():
    m = 200  # 2(2^200) - 1 worlds: enumeration is dead, the quotient is not
    pq, alpha_l = kmm_quotient(m)
    lo, hi = pq.interval()
    assert lo == pytest.approx(kmm_exact_marginal(m), rel=1e-9)
    assert alpha_l == pytest.approx(0.5, abs=1e-9)  # symmetric phases


def test_partial_phase_evidence_gives_certified_interval():
    m = 30
    pq, _ = kmm_quotient(m, side_evidence=(0.6, 0.9))  # 'left-leaning' world
    lo, hi = pq.interval()
    truth_left_heavy = pq.phase_values[0] * 0.75  # any alpha in the box
    assert lo < truth_left_heavy < hi
    assert hi - lo > 0  # weights not identified: honest interval
    # point quotient sits outside this evidence-shifted interval
    assert not (lo <= kmm_exact_marginal(m) <= hi)


def test_residual_mass_pads_upper_only():
    pq = PhaseQuotient([0.4, 0.1], [(0.5, 0.5), (0.4, 0.4)], residual_mass=0.1)
    lo, hi = pq.interval()
    assert lo == pytest.approx(0.4 * 0.5 + 0.1 * 0.4)
    assert hi == pytest.approx(lo + 0.1)


# ---- proof-carrying kernels --------------------------------------------------------

@pytest.fixture(scope="module")
def small_graph():
    import random as _random

    from worldkernel.barrier import random_regular

    return random_regular(10, 3, _random.Random(11))


def test_honest_proof_accepted(small_graph):
    prover = SumcheckProver(small_graph)
    z = prover.claimed_z()
    assert z == int(round(hardcore_z(small_graph)))  # the claim is the truth
    assert verify_z(small_graph, z, prover)


def test_false_claim_rejected(small_graph):
    prover = SumcheckProver(small_graph)
    z = prover.claimed_z()
    assert not verify_z(small_graph, z + 1, prover)
    assert not verify_z(small_graph, 2 * z, prover)


def test_consistent_liar_rejected(small_graph):
    """A prover that forges round 1 to carry a false claim still dies at a
    later round: the recursion pins it against the honest polynomial."""

    class Liar(SumcheckProver):
        def round_poly(self, prefix, var):
            evals = super().round_poly(prefix, var)
            if var == 0:
                evals = [(e + 1) % ((1 << 61) - 1) for e in evals]
            return evals

    liar = Liar(small_graph)
    false_z = (SumcheckProver(small_graph).claimed_z() + 2) % ((1 << 61) - 1)
    assert not verify_z(small_graph, false_z, liar)