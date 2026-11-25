# foldjet-engine 1.0.0

**miniworld-engine is renamed foldjet-engine.** 1.0.0 is miniworld-engine 3.0.12 under the new name: the same code, the same prebuilt kernels (byte for byte), the same judge state. Base: `v3.0.12` (`44a0fad3a`); rename commit `4c3269398`; repository tidy-up (README, docs index, release notes under `docs/release-notes/`, `tools/`) `d2a05ca31`..`6cf75c747`; tag `foldjet-engine-v1.0.0` (the name `v1.0.0` belongs to miniworld-engine 1.0.0 of 2026-08-25). GitHub: `SanggeunParrk/foldjet-engine` (the old `miniworld-engine` URL redirects).

## What changed

- Python package `miniworld_engine` -> `foldjet_engine`; PyPI distribution `miniworld-engine` -> `foldjet-engine`; kernels wheels `miniworld-engine-kernels-<arch>-cu129` -> `foldjet-engine-kernels-<arch>-cu129` (sm90, sm80, sm120, sm100 + sm100-part2), all `==1.0.0`.
- `import miniworld_engine` (and any submodule) still works: `src/miniworld_engine` is an alias that returns the `foldjet_engine` module objects (the same objects, no second copy of any state) and emits one `FutureWarning`.
- Build log prefix `[foldjet-engine build]`.

## What did not change (on purpose)

Everything that enters a prebuilt kernel's key or is compiled into a binary: every C / C++ / CUDA file, the extension and cubin names (`miniworld_*`), the `--miniworld-gencode` / `--miniworld-ptx` flag markers, `-DMINIWORLD_*`, the `MINIWORLD_*` environment variables (three of them are read by the binaries), the `torch.library` namespaces. Also the requirements tables, judge tokens and model / class names (the judge keys on them), and the JIT / header cache directory names (`miniworld_engine_jit`, `miniworld_engine_headers`) so existing caches stay valid. Historical records (release notes, measurement records) are left as written.

## Proof that no kernel changed

- `_prebuild --compare` of this tree against the 3.0.12 release indexes (CPU node, empty JIT cache): sm90 581 / 581, sm80 75 / 75, sm120 73 / 73, sm100 841 / 841 "in both", nothing added or dropped.
- The kernels wheels are the 3.0.12 wheels repacked (`release100/scripts/repack.py`): every payload file keeps its bytes, only the top directory moves to `foldjet_engine/_prebuilt/`; payload digest (path below the top directory + sha256 of every file) equal for all five wheels.
- Judge `owed` (accuracy / speed): h100_sm90, a100_sm80, 5090_sm120, b200_sm100 all 0 / 0.

## Verification

```
079637a7da53c4de116e143b364881e9f073ab654f1674a20ac8b8448e23cb99  foldjet_engine-1.0.0-py3-none-any.whl
2e1482e3937651d9da76b0152c9c5c79a5dd7d028235b62e44c5a8b7ef00d9f9  foldjet_engine_kernels_sm100_cu129-1.0.0-cp312-cp312-manylinux_2_28_x86_64.whl
3c62a8b710de541368c149a4a8f18b58600ec5eed6bd2b045668663ce2403981  foldjet_engine_kernels_sm100_cu129_part2-1.0.0-cp312-cp312-manylinux_2_28_x86_64.whl
300898a56be9de006139c0fd31170c6f5405a91e18d0c34dd1fe42573018aadb  foldjet_engine_kernels_sm120_cu129-1.0.0-cp312-cp312-manylinux_2_28_x86_64.whl
c4e225cbd0cfd8bc046edc11610369d5e4417e3548f98e4ab836ee05f9e788b6  foldjet_engine_kernels_sm80_cu129-1.0.0-cp312-cp312-manylinux_2_28_x86_64.whl
2d4351cd362b060dd70b3583284ef00f69ed218258c40fcaed2bbb4d7e547ccb  foldjet_engine_kernels_sm90_cu129-1.0.0-cp312-cp312-manylinux_2_28_x86_64.whl
82e0376323c9c8d35f4418be4c9cf29570caea40a3a8ea7b1269614ae01a2ca3  foldjet_engine-1.0.0.tar.gz
```

- H100, installed wheels (empty site directory, no repo, no nvcc, empty JIT cache): 581 / 581 sm90 artifacts resolved from `foldjet_engine/_prebuilt`, 0 compiled, JIT cache empty; `import miniworld_engine.kernels._nvcc is foldjet_engine.kernels._nvcc`: True; capture API imports.
- B200, installed wheels (engine + both sm100 wheels; NVIDIA B200, torch 2.13.0+cu129): 841 / 841 sm100 artifacts resolved, 0 compiled, JIT cache empty; ldd of the 76 `.so`: 0 unresolved, 0 CUDA-13 links.
- CPU tests on this tree: all 30 files of tests/cpu passed (e.g. imports 1038, requirements 1194, modules 500, supports_signature 312, init 261 passed / 2 skipped).
- `twine check`: 7 of 7 PASSED.
