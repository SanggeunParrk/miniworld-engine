# Records

Dated findings: each describes what was true when it ran and is **not** updated when the code
changes. Only records that current docs, code or tests cite are kept in the tree. Everything
else — the Sept 17–23 Transition/TriMul studies, MPNN and A6000 investigations, release
verdicts, retired caches, CuTe-era notes — is at tag `archive/docs-20260928`
(`git show archive/docs-20260928:docs/records/README.md` for the old index; older still:
`archive/pre-tidy-20260927`). Removed research code: [experiments-archive.md](experiments-archive.md).

| record | cited by |
|---|---|
| [v220-measurements-20260928.md](v220-measurements-20260928.md) — H100 v2.2.0 TriMul CUDA vs Triton and module comparisons | `docs/gpus/h100.md`, 2.2.0 release |
| [vast-h100-20260927.md](vast-h100-20260927.md) — rented-H100 workflow (historical) | `docs/gpus/h100.md` |
| [a6000-production-audit.md](a6000-production-audit.md) — A6000 final audit (v2.1) | figure notes |
| [cache-coverage-replay-a6000.md](cache-coverage-replay-a6000.md) — lookups the module matrix asks for that the cache does not serve | autotune builder, product standards |
| [a100-internal-gemm-policy.md](a100-internal-gemm-policy.md) — A100 internal GEMM policy | `docs/gpus/a100.md` |
| [pairformer-b200-latency.md](pairformer-b200-latency.md) — Pairformer pair-track latency on B200 | `docs/gpus/b200.md` |
| [model-shape-cleanup-20260916.md](model-shape-cleanup-20260916.md) — model shape policy | README, dispatch cache, sweep page |
| [autotune-space-20260916.json](autotune-space-20260916.json) — autotune space inventory | `tests/autotune/test_compact_search_space.py` |
| [checkpoint-shapes-20260915.json](checkpoint-shapes-20260915.json) — checkpoint constructor census | shape-contract tests, sweep page |
| [tiling-audit.md](tiling-audit.md), [rename-map.tsv](rename-map.tsv) — tile-axis sweep, kernel rename map | autotune docs, capture |
| [local-patches-20260917/](local-patches-20260917.md) — local patch compatibility | CHANGELOG |
| [trimul-sm90-parity.md](trimul-sm90-parity.md) — CuTe SM90 parity (removed in 2.2.0) | `docs/guides/supported.md` |
| [product-plan.md](product-plan.md) — Aug 25 product plan | standards, code comments |
