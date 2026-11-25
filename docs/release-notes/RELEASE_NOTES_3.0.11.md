# miniworld-engine 3.0.11

Base: 3.0.10 (tag `v3.0.10`). Candidate tree: branch `rc-3.0.11` = `main` `7731ef497` + the version bump `58bea41da` (version 3.0.11, the three kernels wheels pinned `==3.0.11`); the notes commit is the tag target.

**What this release is.** Python-only fixes after 3.0.10: a host-overhead cut on every card (the device capability of the per-call backend gates is cached) and the B200 fixes found by the 2026-11-24 FoldForge runs under compile + CUDA graphs. No `.cu` / `.cpp` / `.h` file changed; every prebuilt artifact is byte-identical to 3.0.10 (see "Prebuilt artifacts").

## Changes

1. **Host overhead: the device capability of the per-call backend gates is cached (`b60e3929b`, test fix `fc48ffb6b`).** `kernels._device.capability(device)` returns the same value as `torch.cuda.get_device_capability`, cached per device index; strings, index-less / non-CUDA devices and calls under `torch.compile` tracing go to torch as before. The hot gates of 13 files call it; `dit_variants/cuda/sm80._sms` (the SM count) is lru-cached. No kernel, launch or output change (no `backend_versions` bump).
2. **B200 PWA FP8 off unless `MINIWORLD_PWA_FP8_B200=1` (`a888a8143`).** With the B200 PWA e4m3 route on, RoseTTAFold3 FP8 gave non-finite coordinates when replayed from CUDA graphs (FoldForge F4 under graphs). The route's v-scale cache is now keyed on the weights' data pointers, versions and shapes. Test `tests/cpu/test_pwa_fp8_b200_gate.py`.
3. **B200 token DiT: the bf16 side-stream fork off by default (`4915ee25a`; `MINIWORLD_TOKEN_B200_BF16_FORK=1` turns it on).** With several shapes' sampler graphs alive in one process (FoldForge's per-shape sampler-graph cache), the bf16 fork gave non-finite coordinates (AF3, 27 FoldBench targets in one process: 9 of 27); with the fork off 27 / 27 finite. The fp32 fork is unchanged. Root cause open (docs/b200/fp8.md, "2026-11-24").
4. **Chai-1 FP8 no longer blocked on the B200 (`37f357dbf`).** `fp8.B200_BLOCKED_MODELS` is empty: with the B200 PWA e4m3 route off, Chai-1 FP8 is finite on the B200 (n = 8, F3 / F4 / F5). The blocking mechanism stays (tested with a stand-in model, `tests/cpu/test_fp8_b200_blocked_models.py`).

## Versions (`backend_versions`)

`TokenDiTProtenix`, `TokenDiTChai1` `cuda:b200_sm100` 25 -> 26 (`900d8d355`: the bf16 fork off changes the B200 bf16 bytes). No module `version` changed; no other card's entry changed.

Judge `owed` (accuracy / speed) on this tree: `h100_sm90` 0 / 0, `a100_sm80` 0 / 0, `5090_sm120` 0 / 0, `b200_sm100` 0 / 0 (B200 TokenDiT* re-issued at 26: 502 accuracy + 502 speed cases, 2,008 tokens, RED 0; `34fa953ff`).

## Prebuilt artifacts

`_prebuild` plan on this tree (cluster CPU partition, the 3.0.10 release cache): sm90 581, sm80 75, sm120 73 artifacts, **0 not built**; the (path, sha) pairs of all three indexes equal 3.0.10's. The kernels wheels carry the same files as 3.0.10 under version 3.0.11 (the engine pins `==3.0.11`). B200 (sm_100): still no kernels wheel (source / JIT only).

## Paper data measured on 3.0.10 + these fixes

FoldForge `main` `0859e57`: F1 / F2 FoldBench AF3 on 3.0.10 (H100, bf16 / FP8, 1,522 / 1,522 each; GPU-h 9.969 / 8.244), F3 / F4 (compile + CUDA graphs) / F5 / F9 on H100, RTX 5090, B200 (on these B200 fixes) and A100 (F4 / F5). Engine `benchmarks/paper/PAPER_PREWORK_STATUS.md`.

## Installed-wheel verification (H100)

dist sha256 (`release3011/dist/SHA256SUMS`):

```
83145b2da5f69b3688277620ef3599603bb8f04c9f52902cb92c2da91a7018c6  miniworld_engine-3.0.11-py3-none-any.whl
301c16f1b5b238c6d0e277baf5e543ce2cfabaecfc5f2adece4120d598935878  miniworld_engine_kernels_sm120_cu129-3.0.11-cp312-cp312-manylinux_2_28_x86_64.whl
ad5a177a5a1190be0c6cb0ed0f52a43c447676de54b711dd6e74d0f340395d2e  miniworld_engine_kernels_sm80_cu129-3.0.11-cp312-cp312-manylinux_2_28_x86_64.whl
a70ab3521d4f82fc9e7eea86f9aad44b33f27cb377ceb12f48c8fef2f6e04eee  miniworld_engine_kernels_sm90_cu129-3.0.11-cp312-cp312-manylinux_2_28_x86_64.whl
57b43a0d136f5d72969b1d5d8cdac2ea917766ab420977d27b932702865e0af8  miniworld_engine-3.0.11.tar.gz
```

Wheels extracted into an empty site directory (no repo on the path, no nvcc, no CUTLASS / MathDx, empty JIT cache; Slurm 67163-67165): all 581 indexed sm90 artifacts resolved by the runtime loaders (190 augmented_attention + 226 trimul_inproj + 165 rest), 0 compiled, 0 builds requested, JIT cache empty afterwards; ldd of all 386 `.so` of the three kernels wheels: 0 unresolved, 0 CUDA-13 links; capture API and init presets import. Kernels-wheel manifest names vs 3.0.10: sm90 366 / 366, sm80 75 / 75, sm120 73 / 73, none dropped or added; the 8 carried sm90 files byte-equal. CPU tests (67166): prebuild plan 7, imports 1038, capture API 13, judge 20, FP8 switch 29, modules 500, B200 PWA gate 2, B200 blocked models 5 passed. A100 / RTX 5090: not re-run (kernel files and index hashes identical to 3.0.10, whose installed wheels were verified on both).
