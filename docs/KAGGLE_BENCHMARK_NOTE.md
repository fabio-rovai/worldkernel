# The World Model Arena on Kaggle: a standing multi-model metacognition leaderboard

Paper-2 material. This note records the public Kaggle Community Benchmark built from
the arena, and the first multi-model frontier result. It extends the single-model
frontier audit (POSITION_PAPER.md, PAPER2_DRAFT.md section 6, results inventory #10)
from one model under a private harness to six frontier models on a published,
reproducible leaderboard.

## What it is

A Kaggle Community Benchmark task (Kaggle Benchmarks SDK, `@kbench.task`) that scores
any model on its calibration over non-identified counterfactuals. The honest answer is
the identified interval; a confident point is a measured overclaim. Ground truth is
computed by LP over the response-type polytope and verified offline, so it cannot drift
or be gamed (the trivial [0,1] answer loses on the Winkler width penalty).

- Published task: https://www.kaggle.com/benchmarks/tasks/fabiorovai/world-model-arena-full/1
- Generator: `experiments/kaggle_arena_full.py` (pure LP, scipy)
- Kaggle task: `integrations/kaggle_benchmarks_task.py` (data embedded, no runtime scipy)
- Data: `experiments/kaggle_benchmark_data_full.json` (75 tasks, regenerable, seed 11)

## Design (three LP-verified classes, 75 worlds)

| class | query | identified by | mean width |
| --- | --- | --- | --- |
| two_arm_pn | Probability of Necessity, binary X->Y | Tian-Pearl LP, 8-atom polytope | 0.43 |
| mediation_nde | Natural Direct Effect, X->M->Y | 64-stratum LP, sign undetermined | 0.82 |
| karm_coherence | P(Y(arm1)=1, Y(arm2)=0), 3 arms | joint feasibility LP, 8 types | 0.35 |

Every task is verified at build time: the true counterfactual provably lies inside its
identified interval, else the build aborts. Scoring: coverage, overclaim rate, and the
Winkler interval score at risk a = 0.05; leaderboard value = 1 / (1 + mean Winkler).

## First result (six frontier models, 2026-06-13)

| model | calibration | coverage | overclaim |
| --- | --- | --- | --- |
| DeepSeek-R1 | 0.725 | 0.92 | 0.58 |
| Gemini 3 Flash | 0.667 | 0.86 | 0.58 |
| Gemini 3.1 Pro | 0.651 | 0.96 | 0.44 |
| Grok 4.20 (reasoning) | 0.649 | 0.95 | 0.45 |
| GPT-5.5 | 0.635 | 0.98 | 0.44 |
| Claude Opus 4.8 | 0.193 | 0.71 | 0.67 |

Findings, in order of strength:

1. **No model is calibrated.** Overclaim 0.44 to 0.67 for every model. The single-model
   computation gap from the frontier audit generalises: it is not a Claude artifact.
2. **The strongest flagship is the most overconfident.** Claude Opus 4.8 is worst by
   3-4x, driven entirely by the mediation class (0% coverage, 100% overclaim, Winkler
   15x the leader): it commits to a single NDE number on every world and is wrong every
   time, because no point sits in a sign-undetermined interval. This is the computation
   gap of inventory #10, now sharpened and reproduced across models.
3. **Reasoning is mixed, not a fix.** DeepSeek-R1 (reasoning) leads with the lowest
   Winkler, but Grok-reasoning is mid-pack and in an earlier single-class run reasoning
   did not help. Reasoning sometimes surfaces the non-identifiability; it is not general.
4. **The benchmark discriminates.** A 3.75x spread in the leaderboard value across six
   frontier models.

Caveat for honesty: on an earlier 24-task single-class run, Gemini 3.1 Pro errored on
all rows (no parseable output). On the full 75-task arena it completed normally, so that
failure was transient (API or rate limit), not a capability limit.

## What this adds to paper 2

The arena (section 6) was scored against the kernel and a fixed set of programmatic
contenders, plus a single live model. This makes the arena a *standing public benchmark*:
the kernel reference is the achievable ceiling (100% coverage, 0% overclaim by
construction), and any model's gap to it is now a number on a leaderboard the community
can reproduce and extend. The headline that the strongest flagship is the least
calibrated is a citable empirical claim, not a single-model anecdote.

## TODO for the paper

- Scale to the full five-class, 208-task arena (add finite-sample identification and the
  ontology-structured hard-core class with Weitz self-avoiding-walk certificates).
- Run the full public model line-up, with repeated runs for variance estimates.
- Report the kernel reference row on the same leaderboard as the ceiling.
- Pending: Kaggle Benchmarks Resource Grant for elevated model access and managed compute.
