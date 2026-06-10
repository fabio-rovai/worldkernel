# The proof-carrying world-model entry (PCWE) protocol, v0.1

A world-model answer is not a number. It is a number *and the reason an
agent may trust it*. This document specifies the wire format WorldKernel
emits and verifies, so that any system (an MCP client, another world model,
an offline solver, a market of provers) can exchange kernel entries whose
trust is checkable rather than assumed.

## Entry format

```json
{
  "query": {
    "kind": "occupation_marginal | normalizer | pn | nde | coherence | trajectory_cf",
    "instance": { "edges": [[0,1], ...] },
    "params": { "vertex": 0, "lam": 1.0 }
  },
  "answer": { "lo": 0.241, "hi": 0.241 },
  "certificate": {
    "type": "exact-elimination | weitz | sharp-bounds | lp-identified-set |
             backdoor | phase-quotient | sumcheck",
    "data": { }
  },
  "provenance": { "producer": "worldkernel-0.2.0", "seconds": 0.003 }
}
```

`answer` is always an interval; a point is `lo == hi`. Every certificate
type carries exactly the data a polynomial-time verifier needs:

| type | data | verifier cost | guarantee |
| --- | --- | --- | --- |
| `exact-elimination` | `min_fill_width`, elimination order | re-run: O(n·2^w) | exact value |
| `weitz` | depth | re-run the interval recursion | unconditional containment |
| `sharp-bounds` | marginals used | closed form | sharp identified set |
| `lp-identified-set` | the rung-1/2 record | re-solve the LP | sharp identified set |
| `backdoor` | the set B | check max-deg(G−B) ≤ 5: linear time | FPT-exact decomposition |
| `phase-quotient` | phase values, weight constraints | one LP | certified mixture interval |
| `sumcheck` | claimed Z + interactive prover handle | n rounds, 2 evals each + one direct evaluation | false claims rejected w.p. ≥ 1 − nΔ/2^61 |

Three trust modes, in decreasing order of self-sufficiency:

1. **Recompute** (exact-elimination, weitz, sharp-bounds, lp): the
   certificate is a recipe the verifier can simply re-run in polynomial
   time; trust reduces to trusting your own CPU.
2. **Check** (backdoor, phase-quotient): the certificate is a witness whose
   validity is cheaper to check than the answer was to find; the
   search/verify separation does the work.
3. **Interact** (sumcheck): the value was expensive for SOMEONE (an offline
   solver, a stronger prover); the verifier never repeats that effort and
   still rejects false values with overwhelming probability.

## Rules

- An entry with `lo < hi` is not a weaker answer; it is the claim that the
  width IS the identified freedom. Narrowing it requires a named assumption
  (see `evaluate_assumption`), which must then appear in `provenance`.
- A consumer must not collapse an interval to a point silently. Decision
  layers consume intervals directly (`decide_under_uncertainty`).
- A `sumcheck` entry whose verification fails must be discarded entirely,
  not down-weighted: a single failed round is proof of a false claim or a
  broken prover.

## Reference implementation

The `verify_entry` MCP tool (worldkernel-mcp) verifies modes 1 and 2
directly and drives mode 3 against a prover. `worldkernel.proofs` contains
the sum-check prover/verifier pair; `worldkernel.backdoor.verify_backdoor`
and `worldkernel.tractable.min_fill_order` are the witness checkers.
