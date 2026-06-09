"""worldkernel: a world model as the coupling kernel of admissible possible worlds.

The diagonal of the kernel is what prediction recovers (observational and
interventional marginals). The off-diagonal is the cross-world counterfactual
coupling: the quantity every rung-3 query reads and no rung-1/2 data identify.
This package makes the kernel a first-class, computable object.
"""

from .barrier import d_critical, order_parameter
from .kernel import CouplingKernel, exact_interval, frechet_interval, psd_interval
from .tractable import ring_of_cliques, transfer_marginals, weitz_interval
from .mediation import atom_count, nde_interval, random_reference, rung12_summary
from .witness import (
    TwoWorldKernel,
    frechet_harmed_bounds,
    frechet_pn_bounds,
    witness_pair,
)

__version__ = "0.1.0"

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
    "__version__",
]
