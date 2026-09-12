# What has actually been run

`registry.csv` declares an `arch` per kernel and `pyproject.toml` declares a dependency range.
Neither is evidence. This page is the list of combinations something ran on, and it says
"untested" where nothing did — because a support claim that outruns its measurements is how a
consumer books time on hardware the library has never touched.

Every row here is backed by an artifact in the repo: a device manifest under
`src/miniworld_engine/autotune/manifests/`, or a CI job in `.github/workflows/ci.yml`.

## GPU

A kernel is run at the PRECISIONS it declares (`registry.csv`'s `dtypes`), so a card has a result
per precision and the manifest has a row per (kernel, precision). The registry describes the
current source; the manifests below record historical runs and do not certify every kernel
in the current checkout.

| card | precision | torch | CUDA | triton | Python | result | evidence |
|---|---|---|---|---|---|---|---|
| RTX A6000 (sm86) | bf16 | 2.10.0+cu128 | 12.8 | 3.6.0 | 3.12 | `driven 92, ok 85, failed 7, skipped 6` | `manifests/NVIDIA RTX A6000 (sm86).csv` |
| RTX A6000 (sm86) | fp32 | 2.10.0+cu128 | 12.8 | 3.6.0 | 3.12 | `driven 33, ok 33, failed 0, skipped 2` | same file, `dtype` column |
| RTX A5000 (sm86) | bf16 | 2.10.0+cu128 | 12.8 | 3.6.0 | 3.12 | `driven 101, ok 94, failed 7, skipped 6` | `manifests/NVIDIA RTX A5000 (sm86).csv` |
| RTX A5000 (sm86) | fp32 | — | — | — | — | **not run** | no fp32 pass has been made on this card |

The seven bf16 failures are the newly registered mpnn kernels, from those recorded runs: five are outside the
default 5e-02 band with no `rtol` declared, and two would not launch
because every config the untuned reader offered exceeded this card's shared memory. These
records alone do not establish numerical correctness or successful launch on the current source. The fp32 count
fell from 40 because seven kernels stopped declaring fp32 — `layernorm_*_foldstats` and five
`transition_*` rows are `bf16` only now, so their fp32 records were a precision nothing claims and
`devices.record` dropped them.

Every skip is a kernel whose declared `arch` is above sm86. It is not launched, so it costs nothing
and is not a failure — the manifest says `skipped` with the reason, in its own column, rather than
carrying a stale verdict from before the arch gate existed.

The A5000's seven failures are the same seven the A6000 has, and for the same two reasons -- five
mpnn kernels outside a band nothing declares, two that will not launch on an untuned card. Its six
arch-gated kernels were relabelled from `failed` to `skipped` from the refusal message they already
carried. Its fp32 half has never been run, so read the row as bf16 evidence and nothing more.

`tests/registry/test_the_support_page_counts_its_own_evidence.py` checks every number above against
the manifest it cites, so this table cannot age past its evidence again.

## GPU that has NOT been run

| declared | kernels | ever executed |
|---|---|---|
| sm80 | 109 | yes, on sm86 (which satisfies sm80) |
| sm90 | 7 | **no** |
| sm100 | 4 | **no** |

11 kernels are declared for hardware nothing in this repository has ever run them on. They may
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
against which band — plus a `#provenance` first row carrying the version, the commit, whether the
tree was clean and the date. Commit that file and add the row here; the manifest is the evidence,
its provenance row says what the evidence is *of*, and this table is the index.

A manifest whose `#provenance` says `dirty` describes a working tree, not a commit. It is still
useful to you and it is not evidence for anyone else.
