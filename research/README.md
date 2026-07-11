# research/

Out-of-package kernel R&D: perf probes, autotune/tile sweeps, micro-benchmarks,
and superseded/alternative implementations that are **not** part of the shipped
`miniworld_kernels` package or its public contract.

Kept for provenance and future optimization work; nothing here is imported by
`src/`, the benchmarks, or the tests (verified: zero importers). The paths mirror
their old `src/miniworld_kernels/...` location. To revive one, move it back and
fix its relative imports.
