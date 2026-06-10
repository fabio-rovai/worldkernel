# Score world models on counterfactual honesty

*Position paper draft v0.1 (for arXiv after the WorldKernel paper). Fabio Rovai.*

## The claim

World models are evaluated on fidelity: how well their rollouts match held-out
futures. Fidelity is a rung-1/2 metric. It cannot, even in principle, measure
the property an acting agent needs most: whether the model is honest about the
counterfactual questions its training data do not identify. Two worlds with
identical transition laws can disagree completely about "would the other
action have worked?", and a model trained on either sees the same data. We
propose scoring world models on **counterfactual honesty**: answers are
intervals, scored with a proper interval scoring rule against ground truth
known to the benchmark generator, so that sharp valid intervals win, loose
intervals pay for width, and a point answer pays in full whenever it is wrong.

## The rule

Winkler interval score at level alpha: S([l,u], y) = (u - l) + (2/alpha) *
dist(y, [l,u]). Proper; a point is a zero-width interval. Report at two
levels (alpha = 0.2 and 0.02): the two risk regimes, errors-tolerable and
errors-expensive. The interesting object is the crossover, not one number.

## The benchmark

The World Model Arena (github.com/fabio-rovai/worldkernel): five world
classes whose full laws the generator knows and contenders never see; 208
queries; six algorithmic contenders plus any LLM via the AISI Inspect task.
Headline results so far:

- A sharp-identified-set contender (the kernel) holds 100% coverage and 0%
  overclaim in every class and has the best mean rank at both risk levels.
- The standard predictor (independence coupling / plug-in formulas / belief
  propagation) overclaims on 62-95% of queries with 0% coverage.
- The honest crossover: in one class committing genuinely pays at loose
  alpha and fails at strict alpha. The benchmark is not rigged; it measures
  where honesty starts being worth paying for.

## The frontier audit (live result, 2026-06-09)

A frontier LLM (Claude, via CLI) audited on 14 arena questions under two
harness conditions. On two-arm probability-of-necessity questions, permitting
interval answers lifts coverage from 12% to 88% and cuts overclaim from 75%
to 12%: the model KNOWS the Tian-Pearl bounds, and the standard
point-consuming interface destroys that knowledge (the "harness tax"). On
mediation NDE questions: 0% coverage and 100% overclaim in BOTH conditions;
even when permitted to be honest, the model cannot produce the identified
interval, because that interval is the solution of an LP over the
response-type polytope, a computation, not a fact to recall. (Coverage and
overclaim are the primary audit metrics; raw score magnitudes are
contaminated by response-format parsing artifacts and reported as such.)

## The implication

Honesty about counterfactuals is not a scaling property. The training signal
contains rungs 1-2; the off-diagonal coupling is not in it (this is a
theorem, not an observation). Two consequences:

1. Benchmarks must score interval honesty explicitly, or they reward
   confident hallucination of unidentified quantities.
2. The architecture that passes is hybrid by necessity: a learned model
   proposes; a symbolic kernel computes the identified set, certifies it,
   and refuses to overclaim. The interface between them must transport
   intervals and certificates, not points.

## Reproducibility

Every number above regenerates from `experiments/` in the repository and is
locked by the test suite. The Inspect task accepts any model:

    inspect eval integrations/inspect_arena.py --model <provider/model>
