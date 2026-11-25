# miniworld-engine 3.0.12

Base: 3.0.11 (tag `v3.0.11`). Candidate tree: branch `rc-3.0.12` = `main` `0cec1a267` + `6abf79c53` (version 3.0.12, kernels wheels pinned `==3.0.12`, B200 wheels added, README); the notes commit is the tag target. No source change under `src/` since 3.0.11.

**What this release is.** The first release with **prebuilt B200 (sm_100a) kernels**. Until 3.0.11 the B200 was source / JIT only (kernels compiled on first use). The B200 judge tokens are owed 0 since 2026-11-24, so the B200 is now a release target like the H100, A100 and RTX 5090.

## B200 kernels wheels (new)

- `_prebuild --arch sm100` on this tree (cluster CPU partition, CUDA 12.9 nvcc, host CXX gcc 12.4): **841 artifacts** (765 cubin, 76 ext) from 18 families, all built, 0 failed.
- Staged and stripped as the other arches: 1,682 files, 379 MB uncompressed, about 105 MB compressed -- over PyPI's 100 MiB per-file limit. The set is therefore split into **two wheels** by artifact (an extension's directory or a cubin with its sidecar stays whole; greedy by compressed size): `miniworld-engine-kernels-sm100-cu129` (838 files, 52.9 MB) and `miniworld-engine-kernels-sm100-cu129-part2` (844 files, 53.0 MB), no file in both. Both install into `miniworld_engine/_prebuilt/` like the other kernels wheels; the engine requires both (`==3.0.12`, Linux x86_64, CPython 3.12).
- Paths shared with the sm90 / sm80 / sm120 wheels (arch-independent artifacts: part 1 shares 2, part 2 shares 8 / 4 / 4) have identical bytes.

## Other arches

Unchanged: the sm90 / sm80 / sm120 prebuild indexes equal 3.0.11's (sm90 581, sm80 75, sm120 73 artifacts; manifest names vs 3.0.11: 366 / 75 / 73, none dropped or added); the kernels wheels carry the same files under version 3.0.12.

## Verification

dist sha256:

```
b45cbf2dc837937ba97561c884133fe2938e07fb767bd9a2b392bea8a93be877  miniworld_engine-3.0.12-py3-none-any.whl
59a1438aa6bf781641254d009951a8471f1f867858d805434607e0822a7b97be  miniworld_engine_kernels_sm100_cu129-3.0.12-cp312-cp312-manylinux_2_28_x86_64.whl
a5e75679aac2b6c1c72ac62c5190c9650ee47911ac940bb7e3c2ecfd1cbfc422  miniworld_engine_kernels_sm100_cu129_part2-3.0.12-cp312-cp312-manylinux_2_28_x86_64.whl
bfe41e05728b952701abfa8d3b318df4bd605fca5d70acb9083349f6427cc574  miniworld_engine_kernels_sm120_cu129-3.0.12-cp312-cp312-manylinux_2_28_x86_64.whl
c3c44f5fa8f09324a361ac9962a32eb527af9f8e0ecede6b07ad0f6b69c090b6  miniworld_engine_kernels_sm80_cu129-3.0.12-cp312-cp312-manylinux_2_28_x86_64.whl
332bd1593ac99c2e9420d02abb8097fb834284b733d9e450a1763bcb9461edd5  miniworld_engine_kernels_sm90_cu129-3.0.12-cp312-cp312-manylinux_2_28_x86_64.whl
623c98697f17c8ed03336abeeebfa9d0abb13f6e51fa0f9dd910279baf6e5a90  miniworld_engine-3.0.12.tar.gz
```

- **B200, installed wheels** (engine + both sm100 wheels extracted into an empty site directory; no repo on the path, no nvcc, empty JIT cache; NVIDIA B200, torch 2.13.0+cu129): all **841** indexed sm100 artifacts resolved by the runtime loaders in 9 s, **0 compiled**, 0 builds requested, JIT cache empty afterwards; ldd of the 76 `.so`: 0 unresolved, 0 CUDA-13 links.
- sm90 / sm80 / sm120: kernel files and index hashes identical to 3.0.11, whose installed wheels were verified (3.0.11 notes); not re-run.
- `twine check`: 7 of 7 PASSED; the engine METADATA pins the five kernels wheels `==3.0.12`, no git URLs.
