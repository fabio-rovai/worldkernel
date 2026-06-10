# Submitting the arena to inspect_evals

Goal: the World Model Arena listed in
[UKGovernmentBEIS/inspect_evals](https://github.com/UKGovernmentBEIS/inspect_evals),
so counterfactual honesty becomes a metric in AISI's standard catalogue.

Their contribution requirements (from the repo's CONTRIBUTING):

1. an eval package under `src/inspect_evals/<name>/` with the task,
   a README following their template, and a registry entry in
   `tools/listing.yaml`;
2. results on at least one frontier model with the harness pinned;
3. no network access at eval time (our generator is seeded and offline:
   satisfied by construction).

What maps from this repo:

- `integrations/inspect_arena.py` is the task; it needs only a rename to
  their package layout (`world_model_arena/world_model_arena.py`) and the
  `worldkernel` dependency declared (we are on PyPI-able structure already).
- The README content is the arena section of this repo's README plus the
  scoring-rule definition.
- The frontier-model result: the claude audit
  (`experiments/frontier_audit.py`) provides the headline numbers; a pinned
  Inspect run on one API model is the remaining gap.

Submission is a PR to their repository from a fork. ACTION REQUIRED BY THE
MAINTAINER (deliberate: outward-facing, goes out under your name):

    gh repo fork UKGovernmentBEIS/inspect_evals --clone
    # copy the package per layout above, add listing.yaml entry
    # run their CI locally: make check
    gh pr create --repo UKGovernmentBEIS/inspect_evals

Everything up to the PR is prepared here.
