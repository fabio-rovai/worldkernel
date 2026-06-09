# WorldKernel

**A new kind of world model: not a predictor, a coupling kernel.**

[![CI](https://github.com/fabio-rovai/worldkernel/actions/workflows/ci.yml/badge.svg)](https://github.com/fabio-rovai/worldkernel/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](pyproject.toml)

This repository builds, in public, the reference implementation of the framework in the paper *WorldKernel: A World Model is the Coupling Kernel of Admissible Possible Worlds* (Fabio Rovai, preprint in preparation).

## The idea in one paragraph

Today's "world models" are predictors: they learn the law of what happens. The claim here is that a world model is a strictly larger object, a positive semidefinite **coupling kernel K(T, T')** over admissible possible worlds. The **diagonal** K(T, T) is everything prediction can ever recover: the observational and interventional marginals of each world (rungs 1 and 2 of Pearl's ladder). The **off-diagonal** K(T, T') is the cross-world coupling between potential outcomes: the quantity every genuine counterfactual reads, and the quantity that no amount of rung-1/2 data, and no predictor however large, identifies. A system that holds only the diagonal must collapse worlds that differ counterfactually. A system that holds the kernel separates them exactly. That gap is not philosophy; it is measurable, and this repo measures it.

## Four verified results (all reproducible here)

**1. The off-diagonal witness.** Two structural causal models with identical observational tables and identical interventional tables (ACE = 0.20), but probability of necessity 0.286 vs 0.500. An LLM (`claude -p`) given the rung-1/2 data returns one number and collapses the worlds; handed the off-diagonal, it separates them. The coupling is the load-bearing sufficient statistic.

![Off-diagonal witness](figs/counterfactual_witness.png)

**2. Scale: the mediation interval.** Fix *everything* a randomized mediation experiment (X → M → Y) measures and the Natural Direct Effect is still only identified to the interval **[-0.381, +0.187], which spans zero**: the same experimental record is consistent with the direct effect being harmful or helpful. Only the cross-world coupling decides the sign. Computed exactly by LP over the 64-atom response-type polytope; the LLM baseline commits to one point inside the interval and hides the non-identification.

![Mediation interval](figs/benchmark2_mediation.png)

**3. The barrier.** The kernel's PSD structure is real partial-identifying information: a polynomial-time SDP outer bound on cross-world queries, strictly tighter than Fréchet bounds, computable at k = 40 arms where the exact response-type LP has 2^40 variables. And computability has a sharp edge: the aggregate counterfactual under mutual-exclusion constraints tracks the Sly-Sun hard-core threshold at critical degree d_c = 5.141.

![Barrier sweep](figs/eta_barrier_sweep.png)

**4. Real public trials.** The framework on real data: the Lalonde NSW job-training RCT (n=445), the International Stroke Trial (n=19,435) and Tennessee STAR (3 arms, n=5,789). Each trial identifies the kernel's diagonal; the kernel then computes exactly what rung 3 remains. The IST headline: at 19,435 patients the bootstrap sampling spread on the probability-of-necessity endpoints is 0.013, while the identified interval stays 0.099 wide, a **7.6x gap that no additional data closes**, only an off-diagonal assumption (monotonicity: aspirin never kills a patient who would have survived) does. On STAR's three arms, joint feasibility of one law over all arms tightens the Fréchet lower bound on the cross-world coherence from 0.113 to 0.541: the kernel extracting real information pairwise boxes cannot see.

![Public trials](figs/public_trials.png)

## Quickstart

```bash
pip install -e ".[sdp,plots]"
```

```python
from worldkernel import witness_pair, frechet_pn_bounds

A, B = witness_pair()                 # two worlds, same diagonal
A.observational() == B.observational()  # True : same rung 1
A.ace == B.ace                          # True : same rung 2 (0.20)
A.pn(), B.pn()                          # 0.286 vs 0.500 : rung 3 differs
frechet_pn_bounds(0.5, 0.7)             # (0.286, 0.714) : all rung-1/2 can say
```

```python
from worldkernel import nde_interval, random_reference

lo, hi, *_ = nde_interval(random_reference(seed=0))
# (-0.381, 0.187): the NDE of a fully measured experiment, unidentified, spanning zero
```

```python
from worldkernel import frechet_interval, psd_interval, exact_interval
import numpy as np

d = np.random.default_rng(11).uniform(0.15, 0.85, size=8)   # 8-arm trial marginals
frechet_interval(d)   # marginals-only box
psd_interval(d)       # + kernel PSD constraint: tighter, still poly time
exact_interval(d)     # tight set, 2^8 response types (dies past modest k)
```

Reproduce the figures:

```bash
python experiments/counterfactual_witness.py   # add --llm for the claude baseline
python experiments/mediation_interval.py
python experiments/psd_bounds.py
python experiments/barrier_sweep.py
```

## What is in the package

| Module | Object | Role |
| --- | --- | --- |
| `worldkernel.kernel` | `CouplingKernel`, `frechet_interval`, `psd_interval`, `exact_interval` | The kernel as a PSD second-moment matrix and the three nested bounds on cross-world queries |
| `worldkernel.witness` | `TwoWorldKernel`, `witness_pair` | The minimal two-world witness: PN, PS, PNS, and the canonical verified pair |
| `worldkernel.mediation` | `nde_interval`, `rung12_constraints` | The response-type polytope LP for nested cross-world counterfactuals |
| `worldkernel.barrier` | `order_parameter`, `d_critical`, BP and exact hard-core marginals | Where computing the off-diagonal stops being tractable |

Every paper number above is locked in by the test suite (`pytest`).

## Why this is a new type of world model

The architecture this implies is the opposite of "scale the predictor": the **kernel is the intelligence and the LLM is a frontier sensor**. The symbolic kernel computes rung 3 exactly, exposes non-identification instead of hiding it, and certifies its own bounds; the LLM proposes structure and reads the world. The verified experiments show why the division of labour matters: even when handed the correct coupling, the LLM baseline misexecutes the arithmetic. Restriction (bounded structure, ontology-shaped couplings) is the design constraint that keeps the kernel computable below the Sly-Sun barrier, not a concession.

## Roadmap

Built in public. The full plan is in [ROADMAP.md](ROADMAP.md); every milestone is an open issue.

- **v0.1 (now):** kernel objects, witnesses, bounds, barrier, tests, CI
- **v0.2:** estimation from data (Balke-Pearl LP as primary baseline, finite-sample diagonals, 2-mediator 4096-atom LP)
- **v0.3:** coupling discovery: can a learner plus an LLM teacher discover the assumption (monotonicity, independence) that pins rung 3?
- **v0.4:** the agent loop: kernel as the world model of an acting agent, LLM as sensor
- **v0.5:** structured kernels at scale: ontology-shaped couplings, SDP relaxations beyond pairwise

## Related work by the author

- CIVeX: counterfactual interval verification for agents ([arXiv:2605.09168](https://arxiv.org/abs/2605.09168))
- Event-Graph Substrates for world models ([arXiv:2605.15967](https://arxiv.org/abs/2605.15967))
- Open Ontologies ([arXiv:2605.09184](https://arxiv.org/abs/2605.09184))
- Saturating Scaling Laws ([arXiv:2605.23983](https://arxiv.org/abs/2605.23983))

## Citing

Paper preprint in preparation. Until it is up:

```bibtex
@software{rovai2026worldkernel,
  author = {Rovai, Fabio},
  title  = {WorldKernel: A World Model is the Coupling Kernel of Admissible Possible Worlds},
  year   = {2026},
  url    = {https://github.com/fabio-rovai/worldkernel}
}
```

## License

MIT
