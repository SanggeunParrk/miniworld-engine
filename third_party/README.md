# third_party/

Public kernels used as comparison baselines (E1 / E5 of benchmarks/paper). Tracked files are metadata only (README, INVENTORY, UPSTREAM.json with the pinned revision / sha256,
fetch / build scripts, patches); the upstream trees are gitignored (`upstream/`) and fetched by each entry's `fetch.sh`; nothing here is part of the wheel and nothing is vendored.
Never vendor code with an unclear or incompatible license.

<!-- BEGIN third-party-trimul (triangle-multiplication baselines of E5; branch third-party-trimul) -->
## Triangle-multiplication baselines (E5)

| entry | what | license | used by |
|---|---|---|---|
| `anthropic/` | Anthropic uplifting-biomolecular-modeling kits (`anthropic` backend) | Apache-2.0 | E1 / E2 / E3 |
| `fast_trimul/` | `fast_trimul` 3.0.6: CuTe DSL fused triangle multiplicative update (fp16) | Apache-2.0 | E5 `fast_trimul` |
| `gpumode_trimul/` | arseniivanov/trimul: GPU MODE TriMul competition Triton kernels (A100 / H100 / B200) | MIT | E5 `gpumode` |
| `geometric-nn-ops/` | AMD ROCm/geometric-nn-ops 1.0.0 (Triton): `triangle_multiplicative_update` AND `triangle_attention` (one entry, two adapters) | Apache-2.0 | E5 `geonn`, `gnops` |

Installing on a card: `third_party/<entry>/fetch.sh [TARGET]` (idempotent, sha256 checked); the TriMul adapters look in `$PAPER_EXT_PATH`
(default `~/paper-ext`) and in `third_party/<entry>/upstream`, after the engine environment's site-packages.
<!-- END third-party-trimul -->

<!-- BEGIN third-party-attn (Pairformer-operator baselines of E5; branch third-party-attn) -->
## Pairformer-operator baselines (E5)

Common helpers: `_attn_common.sh` (git / wheel pinning). On a card the attention-side adapters read `$PAPER_EXT_ATTN/<name>` (default `~/paper-ext-attn/<name>`), filled by
`DEST=$PAPER_EXT_ATTN/<name> third_party/<dir>/fetch.sh`. Adapters: `benchmarks/paper/E5_other_kernels/ext_<backend>.py`; all backends are installed by `externals.py` (`run_measure.py --ext <backend>`).

| dir | what | license | E5 backend | covers |
|---|---|---|---|---|
| [geometric-nn-ops](geometric-nn-ops/README.md) | AMD ROCm geometric-nn-ops Triton (triangle attention + triangle multiplicative update) | Apache-2.0 | `gnops`, `geonn` | TriAttn, AttentionPairBias (fwd + bwd); TriMul (fwd only) |
| [openfold3](openfold3/README.md) | OpenFold3 Triton evoformer-attention backend | Apache-2.0 | `of3tri` | TriAttn, AttentionPairBias; fwd + bwd |
| [megafold](megafold/README.md) | MegaFold EvoAttention + fused Transition (Triton) | MIT | `megafold`, `megafold_tr` | TriAttn, AttentionPairBias; Transition; fwd + bwd |
| [bionemo-ir](bionemo-ir/README.md) | NVIDIA BioNeMo Inference Runtime (PyPI wheel; CuTeDSL CUBINs) | Apache-2.0 | `bionemo_ir` | TriAttn, OPM, PWA, Transition; inference |
| [turbopairformer](turbopairformer/README.md) | TurboPairFormer Hopper triangle attention (built locally) | Apache-2.0 | `turbopf` | TriAttn; H100 only |
| [ligo-af3](ligo-af3/README.md) | Ligo AlphaFold3 Triton MSA pair-weighted averaging | Apache-2.0 | `ligo_pwa` | PWA; fwd + bwd (patched) |
| [flashbias](flashbias/README.md) | FlashBias low-rank-bias attention | MIT | - | NOT APPLICABLE (dense pair bias); not connected |
| [trifast](trifast/README.md) | trifast 0.1.13 (record of the pinned install; connected in `externals.py`) | MIT | `trifast` | TriAttn |
| [deepspeed](deepspeed/README.md) | DeepSpeed 0.20.0 DS4Sci Evoformer attention (record; connected in `externals.py`) | Apache-2.0 | `ds4sci` | TriAttn |
<!-- END third-party-attn -->

## Index of every candidate (name, purpose, pinned version, license, where it is used)

The authoritative pins are the `UPSTREAM.json` of each entry (revision or wheel sha256); a short form:

| entry | purpose | pinned version | license | used in |
|---|---|---|---|---|
| `anthropic/` | Anthropic uplifting-biomolecular-modeling kernels (TriAttn, TriMul, ...) | git `f4f62fa6` | Apache-2.0 | E1, E2, E3 (`anthropic` backend; not an E5 candidate) |
| `deepspeed/` | DeepSpeed4Science EvoformerAttention (TriAttn) | deepspeed 0.20.0 (pip, `--target`) | Apache-2.0 | E5 `ds4sci` (graph timing invalid: see `benchmarks/paper/E5_other_kernels/README.md`; eager timing used) |
| `trifast/` | Triton triangle attention | trifast 0.1.13 (wheel sha256 in UPSTREAM.json) | MIT | E5 `trifast` |
| `fast_trimul/` | fused TriMul in CuTe DSL (fp16) | fast_trimul 3.0.6 (wheel sha256) | Apache-2.0 | E5 `fast_trimul` |
| `gpumode_trimul/` | GPU MODE TriMul competition Triton kernels | git `558b9f26` | MIT | E5 `gpumode` |
| `geometric-nn-ops/` | AMD triangle attention + triangle multiplicative update (Triton) | git `c58e24cd` (v1.0.0; attention records from `232ed824`, same package files) | Apache-2.0 | E5 `geonn`, `gnops` |
| `openfold3/` | OpenFold3 Triton Evoformer attention | git `4b59bbf3` | Apache-2.0 | E5 `of3tri` |
| `megafold/` | MegaFold EvoAttention + fused Transition | git `845905e7` | MIT | E5 `megafold`, `megafold_tr` |
| `bionemo-ir/` | NVIDIA BioNeMo Inference Runtime (CuTeDSL CUBINs; inference only) | PyPI wheel (sha256 in UPSTREAM.json) | Apache-2.0 (kernel source not published) | E5 `bionemo_ir` |
| `turbopairformer/` | TurboPairFormer Hopper triangle attention | turbopairformer 0.1.3.post1 sources, built locally | Apache-2.0 | E5 `turbopf` (H100 only) |
| `ligo-af3/` | Ligo AlphaFold3 Triton MSA pair-weighted averaging (patched) | git `621531e4` + `patches/` | Apache-2.0 | E5 `ligo_pwa` |
| `flashbias/` | FlashBias low-rank-bias attention | git `254a8b87` | MIT | none: not applicable to a dense pair bias (not connected) |

## Adding a candidate

1. `third_party/<name>/` with `README.md` (what it is, license, what it covers / does not cover), `UPSTREAM.json` (repository, pinned revision or wheel sha256, license, adapter path, E5 backend name), `fetch.sh` (idempotent, pinned, sha256 checked; use `_attn_common.sh` `git_pin` / `wheel_pin`), `.gitignore` (`upstream/`). Never vendor code with an unclear license.
2. One adapter `benchmarks/paper/E5_other_kernels/ext_<name>.py` (TriMul adapters: `PROBE`, `supports`, `forward`; Pairformer-operator adapters: `install()` calling `ext_attn_common.attach`) and one line in `externals.py` (`TRIMUL_REGISTRY` or `ATTN_REGISTRY`, plus `CLASS_OF`). Nothing under `src/` changes.
3. Measure with `run_measure.py ... --ext <name>`, then run `graph_check.py` for it (a library whose kernels leave the CUDA graph is invalid), `collect.py`, `report.py`.
