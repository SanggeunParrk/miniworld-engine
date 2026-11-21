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
