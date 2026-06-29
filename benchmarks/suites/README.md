# Benchmark Suites

Op-specific benchmark suites live here. Each suite should declare:

- the op or kernel under test,
- the implementation labels,
- correctness metrics,
- timing or memory metrics,
- the output schema consumed by the report renderer.

Temporary probes and design experiments belong in `experiments/`, not here.

## Current Suites

- `triangle_attention.py`: bias-only single-direction and bidirectional triangle
  attention module benchmark.
- `triangle_multiplication_bidirectional.py`: bidirectional triangle
  multiplication module benchmark.
