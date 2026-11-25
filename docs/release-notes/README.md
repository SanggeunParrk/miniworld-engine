# Release notes

One file per release, newest first. The files are historical records and are kept as written: every release before 1.0.0
was published as **miniworld-engine** (package `miniworld_engine`), and its notes use that name, its paths and its branch names.
Dates are the tag dates (`git log -1 v<version>`). Releases before 2.5.9 have no notes file; their history is in git (`git tag -l 'v*'`).

| version | date | headline (from the file) |
|---|---|---|
| [foldjet-engine 1.0.0](RELEASE_NOTES_1.0.0.md) | pending | rename of miniworld-engine 3.0.12; kernels byte-identical, judge state unchanged |
| [3.0.12](RELEASE_NOTES_3.0.12.md) | 2026-11-25 | first prebuilt B200 (sm_100a) kernels, as two wheels (PyPI 100 MiB per-file limit) |
| [3.0.11](RELEASE_NOTES_3.0.11.md) | 2026-11-25 | Python-only fixes: cached device capability in the backend gates; B200 fixes from FoldForge runs |
| [3.0.10](RELEASE_NOTES_3.0.10.md) | 2026-11-24 | RTX 5090: token DiT speed fixes, FP8 routes, Boltz-2 |
| [3.0.9](RELEASE_NOTES_3.0.9.md) | 2026-11-22 | B200 fixes: TriangleAttention NaN / dropped key, fp32 Transition with TF32 off |
| [3.0.8](RELEASE_NOTES_3.0.8.md) | 2026-11-22 | A100 / RTX 5090: token DiT `extra_pair_bias` on the `cuda` backend |
| [3.0.7](RELEASE_NOTES_3.0.7.md) | 2026-11-22 | H100 fp32 Transition on CUDA with TF32 off; opt-in bf16 pair-bias logits for the bias-only token DiT |
| [3.0.6](RELEASE_NOTES_3.0.6.md) | 2026-11-21 | 5090 / A100 Transition overflow repair; long-sequence PWA and bias-only token DiT inference (the file's title still says DRAFT; tag `v3.0.6` exists) |
| [3.0.5](RELEASE_NOTES_3.0.5.md) | 2026-11-20 | small RTX 5090 follow-up (Transition, OuterProductMean) |
| [3.0.4](RELEASE_NOTES_3.0.4.md) | 2026-11-20 | sm_80 family (A100, RTX 5090) speed / correctness fixes; bias-only token DiT logits in fp32 |
| [3.0.3](RELEASE_NOTES_3.0.3.md) | 2026-11-19 | token DiT training byte savings by default; opt-in FP8 training routes |
| [3.0.2](RELEASE_NOTES_3.0.2.md) | 2026-11-19 | FP8 inference connections; e4m3 NaN fix in the TriAttn FP8 PV core |
| [3.0.1](RELEASE_NOTES_3.0.1.md) | 2026-11-18 | H100 bf16 training / inference work |
| [3.0.0](RELEASE_NOTES_3.0.0.md) | 2026-11-18 | opt-in FP8 (e4m3) inference, backend `engine-fp8` (H100) |
| [2.5.14](RELEASE_NOTES_2.5.14.md) | 2026-11-17 | A100 and RTX 5090 ports |
| [2.5.13](RELEASE_NOTES_2.5.13.md) | 2026-11-17 | H100 changes (Transition D 768, TriAttn d 64, ...) |
| [2.5.12](RELEASE_NOTES_2.5.12.md) | 2026-11-16 | H100 bidirectional modules |
| [2.5.11](RELEASE_NOTES_2.5.11.md) | 2026-11-16 | H100 OuterProductMean inference at large MSA depth |
| [2.5.10](RELEASE_NOTES_2.5.10.md) | 2026-11-16 | H100 TriangleAttention inference on 16-channel heads |
| [2.5.9](RELEASE_NOTES_2.5.9.md) | 2026-11-15 | H100 TriAttn / TriMul inference (bit-identical); A100 / 5090 LayerNorm |

## How a release is made (as recorded in the notes)

The notes of 3.0.12 and 1.0.0 record the steps; there is no release script in this repository (the wheel staging / repack
scripts lived in a scratch directory, e.g. `release100/scripts/repack.py`):

1. Version bump in `pyproject.toml` and the `==<version>` pins of the `foldjet-engine-kernels-*-cu129` dependencies.
2. `python -m foldjet_engine.kernels._prebuild --arch <arch>` on a CPU node for sm90a / sm80 / sm120a / sm100a, and
   `--compare` against the previous release's `prebuilt-<arch>.json` (nothing added or dropped unless intended).
3. `python -m foldjet_engine.judge --gpu <card> owed` must be 0 / 0 for every card.
4. CPU tests (`tests/cpu`, one file at a time), `twine check` of every distribution.
5. Installed-wheel check on the card (empty site directory, no repo, no nvcc, empty JIT cache): every artifact resolved, 0 compiled.
6. The notes file here records the sha256 of every distribution; the tag sits on the notes commit.
