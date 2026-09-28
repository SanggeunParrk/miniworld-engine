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
| RTX A6000 (sm86) | bf16 | 2.10.0+cu128 | 12.8 | 3.6.0 | 3.12 | `driven 85, ok 78, failed 7, skipped 0, untested 1` | `manifests/NVIDIA RTX A6000 (sm86).csv` |
| RTX A6000 (sm86) | fp32 | 2.10.0+cu128 | 12.8 | 3.6.0 | 3.12 | `driven 33, ok 33, failed 0, skipped 0` | same file, `dtype` column |
| RTX A5000 (sm86) | bf16 | 2.10.0+cu128 | 12.8 | 3.6.0 | 3.12 | `driven 94, ok 87, failed 7, skipped 0` | `manifests/NVIDIA RTX A5000 (sm86).csv` |
| RTX A5000 (sm86) | fp32 | — | — | — | — | **not run** | no fp32 pass has been made on this card |

v2.2.0 (torch 2.13.0+cu129, triton 3.7.1) has not yet been run on a GPU: every row above is a
record from an earlier version and says nothing about the current dependency set.

The seven bf16 failures are the newly registered mpnn kernels, from those recorded runs: five are outside the
default 5e-02 band with no `rtol` declared, and two would not launch
because every config the untuned reader offered exceeded this card's shared memory. These
records alone do not establish numerical correctness or successful launch on the current source. The fp32 count
fell from 40 because seven kernels stopped declaring fp32 — `layernorm_*_foldstats` and five
`transition_*` rows are `bf16` only now, so their fp32 records were a precision nothing claims and
`devices.record` dropped them.

There are no skips left in these rows. The kernels these runs skipped for declaring an `arch` above
sm86 were the CuTe DSL kernels that v2.2.0 removed, and their manifest rows went with them; the
same release dropped the rows of the Triton kernels that lived in those CuTe files or had no caller
left without them. The A6000's one `untested` bf16 row (`cond_transition_fwd_b2b_saveact_triton`)
has no recorded result.

The A5000's seven failures are the same seven the A6000 has, and for the same two reasons -- five
mpnn kernels outside a band nothing declares, two that will not launch on an untuned card. Its fp32
half has never been run, so read the row as bf16 evidence and nothing more.

`tests/registry/test_the_support_page_counts_its_own_evidence.py` checks every number above against
the manifest it cites, so this table cannot age past its evidence again.

## Incomplete architecture coverage

| declared | kernels | ever executed |
|---|---|---|
| sm80 | 106 | yes, on sm86 (which satisfies sm80) |
| sm90 | 10 | **no**: the H100 manifest lists them `untested` or not at all |
| sm100 | 0 | nothing declares sm100 since v2.2.0 removed the CuTe DSL kernels |

No full-registry manifest has been run on sm90 yet. The three TriMul parity kernels the
[SM90 report](../records/cute/trimul-sm90-parity.md) covered were CuTe kernels and left the registry in
v2.2.0, so that report is history, not evidence for the current source. For paths without
execution evidence, `arch` means "written for", not "verified on".

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
