"""Learning the admissibility structure: constraints from data, validated.

The adoption blocker for every symbolic method is hand-specified structure.
This module closes the loop the architecture already defines: a PROPOSER
(here: a statistical screen over observations; equally: an LLM reading
documentation) proposes constraints; the KERNEL validates them against the
data and certifies what the resulting structure buys (the width certificate,
hence exact off-diagonal computability).

The constraint family is mutual exclusion (disjointness): "class u and
class v are never co-asserted". From a binary assertion matrix X (rows =
observed worlds, columns = classes), a pair is PROPOSED as disjoint when its
observed co-occurrence count is zero AND the expected count under
independence is large enough that zero is surprising:

    expected = n * p_u * p_v >= min_evidence.

This is deliberately a SCREEN, not a proof: absence of co-occurrence in
finite data never proves impossibility. The output therefore carries, per
edge, the evidence level e = n * p_u * p_v (the Poisson-approximate expected
co-count; the miss probability under independence is about exp(-e)), and the
validation step rejects any proposed edge the data outright contradict.

``learn_constraints`` returns the proposed graph plus its width certificate:
the learned structure arrives with its own tractability guarantee attached,
which is the entire point of the ontology-to-width bridge.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .tractable import min_fill_order

__all__ = ["LearnedStructure", "learn_constraints", "sample_worlds"]


@dataclass
class LearnedStructure:
    adj: list[set[int]]
    evidence: dict[tuple[int, int], float]  # expected co-count per learned edge
    width: int
    order: list[int]
    rejected: list[tuple[int, int]] = field(default_factory=list)

    @property
    def n_edges(self) -> int:
        return sum(len(a) for a in self.adj) // 2

    def weakest_edges(self, k: int = 5) -> list[tuple[tuple[int, int], float]]:
        """The learned constraints most likely to be sampling artifacts:
        where a proposer (or an experiment) should look next."""
        return sorted(self.evidence.items(), key=lambda kv: kv[1])[:k]


def learn_constraints(
    X: np.ndarray, min_evidence: float = 3.0
) -> LearnedStructure:
    """Induce a disjointness graph from a binary assertion matrix.

    X has shape (n_observations, n_classes). An edge (u, v) is proposed iff
    the observed co-count is zero and n * p_u * p_v >= min_evidence. Edges
    whose co-count is positive are recorded as rejected proposals (the data
    contradict disjointness outright)."""
    X = np.asarray(X)
    if X.ndim != 2:
        raise ValueError("X must be (n_observations, n_classes)")
    n, k = X.shape
    p = X.mean(axis=0)
    co = X.T.astype(float) @ X.astype(float)  # co-occurrence counts
    adj: list[set[int]] = [set() for _ in range(k)]
    evidence: dict[tuple[int, int], float] = {}
    rejected: list[tuple[int, int]] = []
    for u in range(k):
        for v in range(u + 1, k):
            expected = n * p[u] * p[v]
            if co[u, v] == 0 and expected >= min_evidence:
                adj[u].add(v)
                adj[v].add(u)
                evidence[(u, v)] = float(expected)
            elif co[u, v] > 0 and expected >= min_evidence:
                rejected.append((u, v))
    order, width = min_fill_order(adj)
    return LearnedStructure(
        adj=adj, evidence=evidence, width=width, order=order, rejected=rejected
    )


def sample_worlds(
    adj, n: int, rng: np.random.Generator, lam: float = 1.0
) -> np.ndarray:
    """Sample admissible worlds (independent sets) by Glauber dynamics:
    the data-generating process structure learning is tested against."""
    nv = len(adj)
    state = np.zeros(nv, dtype=int)
    out = np.zeros((n, nv), dtype=int)
    sweeps = 20
    for t in range(n):
        for _ in range(sweeps * nv):
            v = int(rng.integers(nv))
            if state[v] == 1:
                state[v] = 0
            blocked = any(state[w] for w in adj[v])
            if not blocked and rng.random() < lam / (1.0 + lam):
                state[v] = 1
        out[t] = state
    return out
