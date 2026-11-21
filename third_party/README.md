# third_party/

Public kernels used as comparison baselines (E1 / E5 of benchmarks/paper). Tracked files are metadata only (README, UPSTREAM.json, fetch / build scripts, patches);
the upstream trees are gitignored (`upstream/`) and fetched by each `fetch.sh` at the pinned revision. Never vendor code with an unclear or incompatible license.

<!-- BEGIN third-party-attn (Pairformer-operator baselines of E5 ext_attn; branch third-party-attn) -->
## Pairformer-operator baselines (E5 `ext_attn`)

Common helpers: `_attn_common.sh` (git / wheel pinning). On a card the adapters read `$PAPER_EXT_ATTN/<name>` (default `~/paper-ext-attn/<name>`), filled by
`DEST=$PAPER_EXT_ATTN/<name> third_party/<dir>/fetch.sh`. Adapters: `benchmarks/paper/E5_other_kernels/ext_<backend>.py` (registry `ext_attn_registry.py`, run by `run_ext_attn.py`).

| dir | what | license | E5 backend | covers |
|---|---|---|---|---|
| [geometric-nn-ops-attn](geometric-nn-ops-attn/README.md) | AMD ROCm geometric-nn-ops Triton triangle attention | Apache-2.0 | `gnops` | TriAttn, AttentionPairBias; fwd + bwd |
| [openfold3](openfold3/README.md) | OpenFold3 Triton evoformer-attention backend | Apache-2.0 | `of3tri` | TriAttn, AttentionPairBias; fwd + bwd |
| [megafold](megafold/README.md) | MegaFold EvoAttention + fused Transition (Triton) | MIT | `megafold`, `megafold_tr` | TriAttn, AttentionPairBias; Transition; fwd + bwd |
| [bionemo-ir](bionemo-ir/README.md) | NVIDIA BioNeMo Inference Runtime (PyPI wheel; CuTeDSL CUBINs) | Apache-2.0 | `bionemo_ir` | TriAttn, OPM, PWA, Transition; inference |
| [turbopairformer](turbopairformer/README.md) | TurboPairFormer Hopper triangle attention (built locally) | Apache-2.0 | `turbopf` | TriAttn; H100 only |
| [ligo-af3](ligo-af3/README.md) | Ligo AlphaFold3 Triton MSA pair-weighted averaging | Apache-2.0 | `ligo_pwa` | PWA; fwd + bwd (patched) |
| [flashbias](flashbias/README.md) | FlashBias low-rank-bias attention | MIT | - | NOT APPLICABLE (dense pair bias); not connected |
| [trifast](trifast/README.md) | trifast 0.1.13 (record of the pinned install; connected in `externals.py`) | MIT | `trifast` | TriAttn |
| [deepspeed](deepspeed/README.md) | DeepSpeed 0.20.0 DS4Sci Evoformer attention (record; connected in `externals.py`) | Apache-2.0 | `ds4sci` | TriAttn |
<!-- END third-party-attn -->
