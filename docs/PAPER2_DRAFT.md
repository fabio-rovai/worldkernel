# Factoring the Barrier: Certified Access to the World Kernel

*Paper 2 draft, v0.1. Sequel to WorldKernel (arXiv:2606.10934). Drafted in
public in this repository; every number below regenerates from
`experiments/` and is locked by `tests/`.*

## Abstract (draft)

The WorldKernel paper established that a world model is the coupling kernel
of admissible possible worlds and that full restriction-uniform access to
the kernel above the Sly-Sun threshold would collapse NP to RP. This paper
asks what an agent can do about it. We give four constructive escape
hatches, each implemented and verified: (1) kernel access is
fixed-parameter tractable in the size of a uniqueness backdoor, not
governed by degree; (2) queries are affine in phase weights, so
low-phase-rank quotients admit certified polynomial evaluation; (3) values
above the barrier can be consumed safely without being computed, via
sum-check proof-carrying entries; (4) the real-axis wall is the shadow of
the complex zero variety, and certified truncation obeys the zero moat. We
then mount four direct attacks on the hardness assumption itself, each with
a reproducible probe, and show each terminates at a sharper wall: zero-moat
clearance, affine degree invariance, low-width pseudorandomness, and parity
pseudorandomness under isolating slices. Finally we operationalize the
theory as an architecture (structure in, certified verdicts out; decisions
on intervals; assumptions priced, never invented) and evaluate it: a proper
interval scoring rule over five world classes where the kernel is the only
contender with full coverage, and a live audit of a frontier LLM exhibiting
both a harness tax (coverage 12% to 88% when intervals are permitted) and a
computation gap (0% coverage on nested counterfactuals under every
condition). The barrier is not beaten; it is factored, priced, and routed
around exactly where theory permits.

## 1. Introduction

From limitation to architecture. Paper 1 (arXiv:2606.10934) proves the
wall; this paper builds the doors. Thesis sentence: *every legitimate
response to the counting barrier is one of: condition (backdoor), quotient
(phases), verify (proofs), or report the certified interval; and every
attempt to walk through the wall relocates the hardness into a measurable
object.*

## 2. Setting (recap, citing paper 1)

Kernel, diagonal/off-diagonal, Sly-Sun wall at d_c = 5.141, the
restriction-uniform reduction. One page, no new claims.

## 3. Four escape hatches

### 3.1 Backdoor collapse (Theorem)

Kernel access is FPT in b(G) = min{|B| : maxdeg(G-B) <= 5}: conditioning on
the trace over B splits Z and every adaptively restricted marginal into at
most 2^|B| below-threshold residuals; the same B survives every adaptive
restriction. Search/verify separation: finding minimum B is hard, checking
the certificate is linear. Worst-case instances need b(G) = Omega(n), as
they must. Scars are deletion-to-uniqueness certificates. [Implementation:
`worldkernel.backdoor`; hub worlds with degree 9 and |B| = 2, exact vs
enumeration to machine precision.]

### 3.2 Phase-quotient access (Theorem, conditional)

Bounded queries are affine in phase weights; with within-phase values
computable (correlation decay inside a phase) and weights confined to a
convex evidence set, an LP yields a certified interval, a point when
identified. Certified rank-2 instance: K(m,m) at m = 200 (2 * 2^200
worlds), exact in microseconds. Honest scope: the certified decomposition
is an assumption; the glass regime has exponential phase rank and there the
correct output is the interval with the unidentified object named.
[`worldkernel.phases`]

### 3.3 Proof-carrying kernels (Theorem + protocol)

The hard-core normalizer is a hypercube sum of a polynomial with
per-variable degree equal to vertex degree: sum-check applies verbatim.
Exponential prover, polynomial verifier, soundness n * maxdeg / 2^61.
Protocol: entry = interval + certificate, seven certificate types, three
trust modes (recompute, check witness, interact). [`worldkernel.proofs`,
docs/PROOF_CARRYING_PROTOCOL.md, `verify_entry`; honest proofs accepted,
false claims and consistent liars rejected.]

### 3.4 The elastic wall (reframing + measurements)

Sly-Sun is the real-axis shadow of the zero variety of Z; zero-freeness
converts to deterministic approximation (Barvinok, Patel-Regts). Measured:
the naive disk ALWAYS fails at lam = 1 (product of root moduli = 1/i_max);
truncation converges geometrically inside the moat and breaks past it; the
segment clearance is the negative-axis Shearer moat, narrowing with degree.
Honest negative finding, test-locked: no positive-real-part zeros at
enumerable sizes; the lam_c pinching lives beyond n = 20.
[`worldkernel.continuation`]

## 4. Four attacks, four walls

Each attack aims at the NP = RP assumption itself; each probe is in
`experiments/` and reproducible.

| attack | probe | outcome | the sharper wall |
| --- | --- | --- | --- |
| complex continuation | `continuation` tests | disk fails universally; path framing forced | zero moat / condition number |
| affine IDL (isolate, densify, linearize) | `idl_probe.py` | first Macaulay degree IDENTICAL scrambled vs plain, grows with n | degree is an affine invariant |
| nonlinear bending | (theory only) | constant-width opening of Feistel-like covers would break standard crypto | low-width pseudorandomness |
| VV parity (isolate, then parity) | `vv_parity_probe.py` | max intermediate ANF width flat across bucket sizes 0/1/2/3+ (3211/3372/2952/2615) | parity pseudorandomness under isolation |
| localized IDL (adjoin inverses) | `lidl_probe.py` | POSITIVE micro-result verified: one localizer t(1-s)=1 collapses the selector-core obstruction to degree 3 while plain Macaulay needs degree 5; but random localizers over F_p change nothing (r = 0/1/2/4 identical) and rank-greedy selection has no signal (uniform gains) | rational pseudorandomness of isolated zeros: working localizers vanish on the rejected branch, i.e. they ENCODE witness bits; finding the chart is the original search problem |

Highlight for the text (Lemma + verified computation): localization is the
first route that genuinely changes the proof system rather than the
instance. The selector-core no-go dies because substituting s = 1 into
t(1-s) = 1 yields -1 = 0: the hard branch is excised from the chart, not
refuted; and the degree-3 collapse is machine-verified against a hidden
core that plain Macaulay cannot crack below degree 5. The relocation
pattern then becomes precise rather than rhetorical: localizers that work
are partial witnesses (g = 1-s vanishes exactly where s = 1), random
denominators over large fields are units on the whole cube (hence
interpolatable, hence at most a factor-n degree saving), and unaided
selection shows zero gain. Open refinement: the rank-gain objective for
'proof scars' is non-discriminating (every localizer contributes the same
block rank); the right objective is damage to the x-sector nullspace.

Section thesis: every route relocates the hardness into a measurable object
rather than removing it, exactly as NP != RP predicts; each probe is a
standing falsification target. The localized route sharpens this from
slogan to mechanism: the wall is not that charts do not help, it is that
helpful charts are made of witness bits.

## 5. The architecture

One object (`WorldModel`): structure + data + assumptions in, Verdicts out
(interval, engine, exactness, diagnostics). Estimation with simultaneous
coverage (corner-evaluated confidence boxes, exact by monotonicity);
sequential dynamics (per-step coupling boxes, exact trajectory bounds,
MC-validated); decisions on intervals (maximin, minimax regret, dominance,
value of information); assumptions validated and priced, never invented;
structure learned from data arrives with its width certificate. Continuous
outcomes via Makarov bounds and optimal-transport coupling extremes. The
LLM's place is outside the object: proposer and sensor, never calculator.

## 6. Evaluation: the arena and the frontier audit

Absorbs docs/POSITION_PAPER.md. The arena (five world classes, six
contenders, Winkler scoring at two risk levels): the kernel is the only
contender in all classes, 100% coverage, 0% overclaim, best mean rank in
both regimes; the predictor overclaims on 62-95% of queries; the honest
k-arm crossover shows where committing stops paying. Real data: NSW, IST
(sampling 0.013 vs identification 0.099: the 7.6x gap data never closes),
STAR, UCI Bank Marketing (PN floor 0.80 with both point estimates at the
floor). Platform test: identical training data, rung-3 truths 1.000 vs
0.298 (stable-worldmodel TwoRoom). Frontier audit (Claude, live): harness
tax (PN coverage 12% to 88% when intervals permitted) and computation gap
(NDE 0% coverage, 100% overclaim, both conditions).

## 7. Related work

Tian-Pearl, Manski, Balke-Pearl (partial identification); Weitz, Sly-Sun,
Barvinok, Patel-Regts (computation); Valiant-Vazirani, Toda (attacks);
backdoors in SAT (Williams-Gomes-Selman lineage; here: backdoors to
uniqueness); LFKN sum-check; DINO-WM/LeWM and the predictive world-model
line; AISI Inspect for the evaluation harness.

## 8. Discussion

What remains genuinely open: ingesting real OWL ontologies as width
certificates; the bent rank-collapse lemma (priced as crypto-implausible);
the positive-axis zero pinching beyond enumerable sizes; Lasserre-2.

---

## Results inventory (claim -> artifact -> lock)

| # | claim | experiment | test | key numbers |
| --- | --- | --- | --- | --- |
| 1 | backdoor FPT, degree 9 world exact via \|B\|=2 | (module demo) | test_escapes.py | machine-zero vs enumeration |
| 2 | K(200,200) quotient exact | (module demo) | test_escapes.py | microseconds, 2*2^200 worlds |
| 3 | sum-check verifier rejects liars | (module demo) | test_escapes.py | soundness n*deg/2^61 |
| 4 | disk always fails; truncation obeys moat | (module demo) | test_continuation.py | min root modulus < 1 always |
| 5 | IDL degree is affine-invariant | idl_probe.py | (probe) | 4,5,4,4 = 4,5,4,4 |
| 6 | parity width blind to isolation | vv_parity_probe.py | (probe) | 3211/3372/2952/2615 |
| 7 | arena: kernel only full-coverage contender | arena.py | test_arena.py | rank 1.60/1.40, overclaim 0% |
| 8 | IST sampling vs identification | public_trials.py | test_public_trials.py | 0.013 vs 0.099 (7.6x) |
| 9 | platform witness | swm_witness.py | (audited) | 1.000 vs 0.298 |
| 10 | frontier harness tax + computation gap | frontier_audit.py | (live audit) | PN 12->88%; NDE 0%, both |
| 11 | trajectory cf bounds exact | (module) | test_dynamics.py | 60k-episode MC validated |
| 12 | coverage-guaranteed estimation | (module) | test_estimate.py | simulated coverage >= nominal |
| 13 | ontology width tracks branching | ontology_width.py | test_width.py | 1,110 classes, 1.3 s |
| 14 | bank marketing PN floor | public_trials.py | (merged branch) | PN in [0.80, 1.00] |
