"""worldkernel: a world model as the coupling kernel of admissible possible worlds.

The diagonal of the kernel is what prediction recovers (observational and
interventional marginals). The off-diagonal is the cross-world counterfactual
coupling: the quantity every rung-3 query reads and no rung-1/2 data identify.
This package makes the kernel a first-class, computable object.
"""

from .backdoor import (
    backdoor_marginal,
    backdoor_z,
    find_backdoor,
    hub_world,
    verify_backdoor,
)
from .barrier import d_critical, order_parameter
from .continuation import indep_poly, log_taylor_estimate, shearer_radius, zero_moat
from .continuous import (
    abs_effect_bounds,
    comonotone_qte,
    effect_quantile_bounds,
    makarov_bounds,
    prob_benefit_bounds,
    supermodular_extremes,
)
from .learn import LearnedStructure, learn_constraints, sample_worlds
from .decide import Decision, decide
from .dynamics import (
    CorridorWorld,
    counterfactual_success_interval,
    independence_point,
)
from .estimate import (
    ace_from_counts,
    harmed_bounds_from_counts,
    pn_bounds_from_counts,
)
from .interaction import (
    interaction_rank,
    ring_clique_marginal,
    ring_clique_pair,
    treewidth_cost,
)
from .kernel import CouplingKernel, exact_interval, frechet_interval, psd_interval
from .model import Verdict, WorldModel
from .phases import PhaseQuotient, kmm_exact_marginal, kmm_quotient
from .proofs import SumcheckProver, verify_z
from .propose import evaluate as evaluate_assumption
from .query_algebra import (
    AlgebraVerdict,
    expected_occupancy,
    linear_query,
    pairwise_coherence,
    ratio_query,
)
from .query_class import (
    QueryVerdict,
    coupling_rank,
    necessity_from_couplings,
    occupation_pattern_prob,
    pairwise_offdiagonal,
)
from .query_scar import (
    ScarQuery,
    kmm_marginal_via_scar,
    local_query_via_scar,
    shiraishi_mori_block,
)
from .tractable import ring_of_cliques, transfer_marginals, weitz_interval
from .mediation import atom_count, nde_interval, random_reference, rung12_summary
from .witness import (
    TwoWorldKernel,
    frechet_harmed_bounds,
    frechet_pn_bounds,
    witness_pair,
)

__version__ = "0.2.0"

__all__ = [
    "CouplingKernel",
    "TwoWorldKernel",
    "witness_pair",
    "frechet_pn_bounds",
    "frechet_harmed_bounds",
    "frechet_interval",
    "psd_interval",
    "exact_interval",
    "nde_interval",
    "rung12_summary",
    "random_reference",
    "atom_count",
    "order_parameter",
    "d_critical",
    "weitz_interval",
    "ring_of_cliques",
    "transfer_marginals",
    "WorldModel",
    "Verdict",
    "decide",
    "Decision",
    "evaluate_assumption",
    "pn_bounds_from_counts",
    "harmed_bounds_from_counts",
    "ace_from_counts",
    "counterfactual_success_interval",
    "independence_point",
    "CorridorWorld",
    "find_backdoor",
    "verify_backdoor",
    "backdoor_z",
    "backdoor_marginal",
    "hub_world",
    "PhaseQuotient",
    "kmm_quotient",
    "kmm_exact_marginal",
    "SumcheckProver",
    "verify_z",
    "indep_poly",
    "zero_moat",
    "shearer_radius",
    "log_taylor_estimate",
    "makarov_bounds",
    "prob_benefit_bounds",
    "effect_quantile_bounds",
    "comonotone_qte",
    "abs_effect_bounds",
    "supermodular_extremes",
    "learn_constraints",
    "LearnedStructure",
    "sample_worlds",
    "ScarQuery",
    "kmm_marginal_via_scar",
    "local_query_via_scar",
    "shiraishi_mori_block",
    "QueryVerdict",
    "pairwise_offdiagonal",
    "occupation_pattern_prob",
    "coupling_rank",
    "necessity_from_couplings",
    "AlgebraVerdict",
    "expected_occupancy",
    "pairwise_coherence",
    "linear_query",
    "ratio_query",
    "ring_clique_pair",
    "ring_clique_marginal",
    "interaction_rank",
    "treewidth_cost",
    "__version__",
]
