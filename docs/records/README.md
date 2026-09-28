# Records

Dated findings: each describes what was true when it ran and is **not** updated when the code
changes. Only records that current docs, code or tests cite are kept in the tree. Everything
else — the Sept 17–23 Transition/TriMul studies, MPNN and A6000 investigations, release
verdicts, retired caches, CuTe-era notes — is at tag `archive/docs-20260928`
(`git show archive/docs-20260928:docs/records/README.md` for the old index; older still:
`archive/pre-tidy-20260927`). Removed research code: [experiments-archive.md](experiments-archive.md).

| record | cited by |
|---|---|
| [h100/v220-measurements-20260928.md](h100/v220-measurements-20260928.md) — H100 v2.2.0 TriMul CUDA vs Triton and module comparisons | `docs/status/h100.md`, 2.2.0 release |
| [h100/vast-h100-20260927.md](h100/vast-h100-20260927.md) — rented-H100 workflow (historical) | `docs/gpus/h100.md` |
| [ampere/a6000-production-audit.md](ampere/a6000-production-audit.md) — A6000 final audit (v2.1) | figure notes |
| [ampere/cache-coverage-replay-a6000.md](ampere/cache-coverage-replay-a6000.md) — lookups the module matrix asks for that the cache does not serve | autotune builder, product standards |
| [ampere/a100-internal-gemm-policy.md](ampere/a100-internal-gemm-policy.md) — A100 internal GEMM policy | `docs/status/a100.md` |
| [pairformer/pairformer-b200-latency.md](pairformer/pairformer-b200-latency.md) — Pairformer pair-track latency on B200 | `docs/status/b200.md` |
| [reports/model-shape-cleanup-20260916.md](reports/model-shape-cleanup-20260916.md) — model shape policy | README, dispatch cache, sweep page |
| [reports/autotune-space-20260916.json](reports/autotune-space-20260916.json) — autotune space inventory | `tests/autotune/test_compact_search_space.py` |
| [model-shapes/checkpoint-shapes-20260915.json](model-shapes/checkpoint-shapes-20260915.json) — checkpoint constructor census | shape-contract tests, sweep page |
| [audits/tiling-audit.md](audits/tiling-audit.md), [audits/rename-map.tsv](audits/rename-map.tsv) — tile-axis sweep, kernel rename map | autotune docs, capture |
| [cache/local-patches-20260917/](cache/local-patches-20260917/README.md) — local patch compatibility | CHANGELOG |
| [cute/trimul-sm90-parity.md](cute/trimul-sm90-parity.md) — CuTe SM90 parity (removed in 2.2.0) | `docs/getting-started/supported.md` |
| [development/product-plan.md](development/product-plan.md) — Aug 25 product plan | standards, code comments |
