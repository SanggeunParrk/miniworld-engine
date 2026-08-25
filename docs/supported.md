# What has actually been run

`registry.csv` declares an `arch` per kernel and `pyproject.toml` declares a dependency range.
Neither is evidence. This page is the list of combinations something ran on, and it says
"untested" where nothing did — because a support claim that outruns its measurements is how a
consumer books time on hardware the library has never touched.

Every row here is backed by an artifact in the repo: a device manifest under
`src/miniworld_engine/autotune/manifests/`, or a CI job in `.github/workflows/ci.yml`.

## GPU

| card | arch | torch | CUDA | triton | Python | result | evidence |
|---|---|---|---|---|---|---|---|
| RTX A6000 | sm86 | 2.10.0+cu128 | 12.8 | 3.6.0 | 3.12 | `driven 94, ok 94, failed 0, skipped 9` | `manifests/NVIDIA RTX A6000 (sm86).csv` |
| RTX A5000 | sm86 | 2.10.0+cu128 | 12.8 | 3.6.0 | 3.12 | same shape | `manifests/NVIDIA RTX A5000 (sm86).csv` |

The nine skips are kernels whose declared `arch` is above sm86; they are not launched, so they
cost nothing and cannot be reported as failures.

## GPU that has NOT been run

| declared | kernels | ever executed |
|---|---|---|
| sm80 | 94 | yes, on sm86 (which satisfies sm80) |
| sm90 | 2 | **no** |
| sm100 | 7 | **no** |

Nine kernels are declared for hardware nothing in this repository has ever run them on. They may
work; the point is that nobody knows, and `arch` should be read as "written for", not "verified
on", until a manifest for that card exists here.

## CPU

| Python | what runs | evidence |
|---|---|---|
| 3.12 | ruff, ty, the whole non-GPU suite, the wheel build and an isolated import | `checks` + `wheel` jobs |
| 3.10 | the non-GPU suite (the declared `requires-python` floor) | `floor` job |

## Dependencies

`torch>=2.8` and `triton>=3.3` are the declared floors. The triton floor is from code, not from a
test: `autotune/cache.py` prefers `triton.backends.nvidia.compiler.get_ptxas_version`, which
exists from 3.3, and falls back below it. **Nothing has been run on any torch or triton older than
the row above**, so treat the floors as "the API is there", not "this was measured".

`einops`, `jaxtyping` and `numpy` carry no floor. Adding one would be inventing a number: no
version of this repository has been exercised against an older release of any of them, and a
guessed floor reads like evidence. What is true is the row above.

## How to add a row

Run `python -m miniworld_engine.autotune.run_all` on the card. It writes
`autotune/manifests/<device>.csv` with one line per kernel — what ran, what it measured, and
against which band. Commit that file and add the row here; the manifest is the evidence and this
table is its index.
