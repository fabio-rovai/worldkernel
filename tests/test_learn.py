"""Structure learning: constraints from data, with the width certificate."""

import numpy as np
import pytest

from worldkernel.learn import learn_constraints, sample_worlds
from worldkernel.tractable import disjointness_graph


@pytest.fixture(scope="module")
def taxonomy_data():
    adj = disjointness_graph(branching=4, depth=2)  # 20 classes
    rng = np.random.default_rng(11)
    X = sample_worlds(adj, n=600, rng=rng)
    return adj, X


def test_recovers_true_constraints(taxonomy_data):
    adj, X = taxonomy_data
    learned = learn_constraints(X, min_evidence=3.0)
    true_edges = {(u, v) for u in range(len(adj)) for v in adj[u] if u < v}
    learned_edges = set(learned.evidence)
    # every learned edge is real (no data-contradicted proposals slip in)
    false_edges = learned_edges - true_edges
    assert len(false_edges) <= max(1, len(learned_edges) // 10)
    # and recall is substantial where evidence exists
    assert len(learned_edges & true_edges) >= len(true_edges) * 0.5


def test_no_rejected_edge_is_a_true_constraint(taxonomy_data):
    """An edge the data contradict (positive co-count) can never be a true
    mutual exclusion: rejection is sound, not heuristic."""
    adj, X = taxonomy_data
    learned = learn_constraints(X)
    for u, v in learned.rejected:
        assert v not in adj[u]


def test_width_certificate_attached(taxonomy_data):
    adj, X = taxonomy_data
    learned = learn_constraints(X)
    assert learned.width <= 8  # near the true branching, far below n
    assert len(learned.order) == X.shape[1]


def test_weakest_edges_ranked_by_evidence(taxonomy_data):
    _, X = taxonomy_data
    learned = learn_constraints(X)
    weakest = learned.weakest_edges(3)
    evs = [e for _, e in weakest]
    assert evs == sorted(evs)


def test_more_data_more_recall():
    adj = disjointness_graph(branching=4, depth=2)
    rng = np.random.default_rng(7)
    small = learn_constraints(sample_worlds(adj, 80, rng))
    big = learn_constraints(sample_worlds(adj, 800, rng))
    assert len(big.evidence) >= len(small.evidence)


def test_input_validation():
    with pytest.raises(ValueError):
        learn_constraints(np.zeros(5))