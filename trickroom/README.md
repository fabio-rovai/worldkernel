# Trick Room: the kernel thesis as a working planner

Inherited from a private research fork of [galilai-group/stable-worldmodel](https://github.com/galilai-group/stable-worldmodel) (MIT), now archived; this directory preserves the project's own additions, results, and audit trail. It is the WorldKernel thesis run as a head-to-head planning experiment: a **specification-derived symbolic substrate** (typed-RDF ontology rules compiled to a physics engine, zero training trajectories) plugged into stable-worldmodel's own CEM-MPC harness via its `Costable.get_cost` protocol, against a **pretrained latent world model** (LeWM, 1,000 trajectories) on Two-Room planning.

## Headline result (sequential evaluation, n = 200 sweep)

| Planner | Success | 95% CI |
| --- | --- | --- |
| Substrate (correct door prior) | **82%** | 69-90% |
| Substrate (no door prior) | 66% | 52-78% |
| LeWM (1,000 training trajectories) | 48% | 35-61% |

Substrate same-room success is 100% regardless of prior (LeWM: 62%); cross-room depends on the door prior (19% wrong vs 57% correct, LeWM 29%). The substrate also shows structurally smaller and more predictable out-of-distribution degradation (see `results/ood_*.json`).

The earlier "100% vs 66%" figure was an evaluation artefact (auto-mode environment recycling); [ARGUMENT.md](ARGUMENT.md) documents all three artefacts found and corrected. The numbers above are the post-audit ones.

Experiment C (hybrid gating, clean run, seeds 92-291): a spatially gated residual correction (15 px) improves cross-room success by +16.8pp (p = 0.001) at no significant same-room cost, while naive ungated hybridization collapses same-room performance by 36.8pp; see `results/experiment_c_hybrid.json`. The lesson mirrors the kernel architecture: corrections must be gated by structure, not blended in.

## Why it lives in this repository

The substrate is an ontology-specified world model: `tbox/tworoom.ttl` (5 classes, 3 rules) compiles to the planner's dynamics. That is the ontology-to-width bridge of `worldkernel.tractable` in working form: the specification *is* the world model, the structure *is* the tractability guarantee, and no gradient ever flows. The kernel argues prediction misses the off-diagonal; Trick Room shows what replaces it: write the world down, compile it, plan in it.

## Layout

- `substrate/`: `SubstrateCostModel` and engines: `python_substrate.py` (self-contained reference engine), `rust_model.py` (JSON-RPC bridge), `pyo3_model.py` (PyO3 bridge), `tbox_compiler.py` / `code_compiler.py` (ontology to dynamics), `smooth_substrate.py`, `hybrid_substrate.py`. Drop into `stable_worldmodel/wm/substrate/` of a stable-worldmodel checkout to run.
- `tbox/tworoom.ttl`: the Two-Room ontology.
- `scripts/`: benchmark drivers (`trick_room_two_room*.py`, `lewm_two_room.py`, `ood_*.py`, `counterfactual_two_room.py`), `nl_to_substrate.py` (natural-language to substrate spec), sweep and plotting utilities, planner config.
- `results/`: every JSON behind the tables above, including the per-episode decompositions and the artefact-audit reruns.
- `tests/`: substrate engine and probe tests (run inside a stable-worldmodel checkout).
- `ARGUMENT.md`: the full argument skeleton and evaluation audit, kept verbatim as the honest record.

## Running

The Python engine is self-contained given `pip install stable-worldmodel torch`. The Rust engine (the headline-throughput path) lives in a separate private repository: set `SUBSTRATE_RUST_DIR` to its checkout and `cargo build --release --bin tworoom_substrate_rpc`; `python_substrate.py` reproduces the same dynamics without it.
