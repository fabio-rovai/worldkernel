# Where certificates beat fidelity: deployment verticals

The strategic claim: frontier world models compete on fidelity of generated
futures; WorldKernel competes on *guarantees about counterfactuals*. There
are domains where the second property is the purchasable one, because a
regulator, an auditor, or a liability regime demands "show me the model
knows what it does not know". This note names them and what the kernel
ships into each.

## 1. Clinical decision support

The question a clinician actually asks is rung-3: "would this patient have
recovered without the drug?" (probability of necessity), "who is harmed?"
(fraction harmed). Trials identify neither; they identify the kernel
diagonal. The package already demonstrates the full pipeline on real trial
data (IST, n=19,435): counts in, simultaneous-coverage intervals out,
sampling inflation separated from identification freedom, assumptions priced
explicitly. Regulatory hook: software-as-a-medical-device frameworks
increasingly require uncertainty characterization; an interval with a
coverage guarantee is auditable, a point estimate is not.

## 2. Financial risk and conduct

Counterfactual questions are the substance of conduct regulation: "would
this customer have been harmed under the alternative product?" is a
fraction-harmed query, unidentified from observational books. The kernel's
contribution is to expose that non-identification instead of hiding it, and
to price the assumption that closes it. Decision rules over intervals
(maximin, minimax regret, dominance) map directly onto risk-appetite
statements.

## 3. Autonomous-systems assurance

A planner's safety case needs "would the alternative action have avoided the
incident?": exactly the trajectory counterfactual the dynamics module bounds
exactly, with the predictor's blind spot (its answer provably cannot
condition on the episode's own evidence) as the contrast. The
proof-carrying entry protocol (docs/PROOF_CARRYING_PROTOCOL.md) is the
artifact an assurance case can file: value + certificate, third-party
checkable.

## 4. Government and evaluation ecosystems

The arena is packaged as an AISI Inspect task; counterfactual-honesty
scoring slots into the same machinery used for frontier-model evaluations.
The pitch to an evaluation body is not "use our model"; it is "add this
metric": coverage, overclaim rate, and Winkler scores at two risk levels for
any world model or LLM consumed as one.

## What is deliberately out of scope

Training foundation models, competing on rollout fidelity, or any vertical
whose buyer wants plausible futures rather than certified counterfactuals.
The moment the roadmap requires pretraining, the strategy has failed.
