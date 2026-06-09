# Contributing

This project is built in public and contributions are welcome.

## Ground rules

- Every claim lands with a test. If you add a bound, add the instance that shows it is tight (or the check that it is valid).
- Numbers in the README come from `experiments/` scripts and are locked by `tests/`. If you change one, change all three together.
- The kernel is the calculator. PRs that move rung-3 arithmetic into an LLM call will be declined; PRs that use an LLM to *propose* structure the kernel then verifies are exactly the point.

## Setup

```bash
git clone https://github.com/fabio-rovai/worldkernel
cd worldkernel
pip install -e ".[dev,plots]"
pytest
```

## Good first issues

See the issues labelled `good-first-issue`, or pick any unchecked box in [ROADMAP.md](ROADMAP.md) and open an issue to claim it.
