# E5: other public kernels

External backends of the measurer, installed at run time by `externals.py` (`run_measure.py ... --ext ds4sci trifast`; nothing under `src/`
changes): the module keeps its own LayerNorm, projections, gate, out-projection and residual; only the attention core is the library's
(the same split as `cueq` / `cuda`). Measured on the E2 grid's TriAttn cases (inference and training), next to `pytorch`, `cueq` and the engine's `cuda`;
a case a library does not serve is `unsupported` with its reason (a missing package, dtype, head width, a forward error: recorded, never re-routed).

| backend | library | notes |
|---|---|---|
| `ds4sci` | DeepSpeed4Science `DS4Sci_EvoformerAttention` (`deepspeed.ops.deepspeed4science`) | bf16 / fp16; JIT-builds its CUTLASS extension (needs `CUTLASS_PATH`, nvcc, ninja); q scaled by `d^-0.5` before the call; key mask and pair bias passed as the two biases |
| `trifast` | `trifast.triangle_attention` (`pip install trifast`, latkins/trifast 0.1.13) | Triton kernels, fp32 / bf16 / fp16; q / k / v `[B, H, N, N, d]`, bias `[B, H, N, N]`, mask `[B, N, N]` (True = masked), softmax scale inside; forward + backward |

Triangle multiplication: no further published fused kernel was found besides cuEquivariance (`cueq`) and the Anthropic kits, both already
backends of E1 / E2; E5 therefore covers TriAttn only (`externals.tm_candidates()`).

**Install** (into a separate path, never the engine env; versions are recorded by `pip freeze` into `ext_versions.txt` by hand when installing):

    pip install --target $HOME/paper-ext --no-deps trifast platformdirs psutil hjson py-cpuinfo ninja msgpack
    pip install --target $HOME/paper-ext --no-deps deepspeed          # 0.20.0 at the time of writing; the op JIT-builds on first use (needs CUTLASS_PATH, nvcc, ninja)

`externals.install` appends `$PAPER_EXT_PATH` (default `~/paper-ext`) to `sys.path` (after the engine environment, so it never shadows an engine dependency; do not install pydantic or other packages the engine env already has). Cards: ds4sci's CUTLASS kernels target sm80+ (A100, H100; sm120
untested), trifast is Triton (any card).

    cd benchmarks/paper && REP=1 paper_common/chain.sh 6 E5_other_kernels/h100.sbatch ; python E5_other_kernels/collect.py --card h100

Output `E5_<card>.csv` (columns as E1). Status: see `../README.md` (adapters smoke-run; library versions to be pinned at the campaign).

## ext_attn: the Pairformer-operator candidates of third_party/ (branch third-party-attn)

Further public kernels as external backends of the measurer, one adapter file each (`ext_<name>.py`), registered by one line in `ext_attn_registry.py`; nothing under `src/` changes.
Run through `run_ext_attn.py` (it is `paper_common/run_measure.py` with the registry's names accepted by `--ext`; the coordinator merges the registry into `externals.py` later):

    python E5_other_kernels/run_ext_attn.py E5_other_kernels/cfg_ext_attn.yaml --kind speed --out E5_other_kernels/records/<card>/ext_attn.r1.jsonl --ext gnops of3tri megafold bionemo_ir turbopf ligo_pwa megafold_tr
    python E5_other_kernels/run_ext_attn.py E5_other_kernels/cfg_ext_attn_acc.yaml --kind accuracy --out ... --ext ...     # vs the float64 gold of the module's reference

The upstream trees are installed per card by `third_party/<name>/fetch.sh` into `$PAPER_EXT_ATTN` (default `~/paper-ext-attn`; not `~/paper-ext`, which the trifast / ds4sci campaigns use).
Cases: `requirements_ext_attn.csv` (AF3 setup rows only, L 512 | 1024 | 2048); a backend serves only the classes its adapter attaches to,
and a case outside a library's range is recorded `unsupported` with the reason (never re-routed).

| backend | library (license) | classes | modes / dtypes | cards |
|---|---|---|---|---|
| `gnops` | ROCm geometric-nn-ops Triton (Apache-2.0) | TriangleAttention, AttentionPairBias | fwd + bwd, bf16 / fp16 / fp32 | all |
| `of3tri` | OpenFold3 Triton evoformer attention (Apache-2.0) | TriangleAttention, AttentionPairBias | fwd + bwd, bf16 / fp16 / fp32 | all |
| `megafold` | MegaFold EvoAttention Triton (MIT) | TriangleAttention, AttentionPairBias | fwd + bwd | all |
| `bionemo_ir` | NVIDIA BioNeMo Inference Runtime (Apache-2.0, CuTeDSL CUBIN) | TriangleAttention, OuterProductMean, MSAPairWeightedAveraging, Transition | inference, bf16 / fp16 | A100, H100 (5090: cuEquivariance / Triton fallbacks) |
| `turbopf` | TurboPairFormer (Apache-2.0, CUDA, built locally) | TriangleAttention | fwd + bwd, bf16, d <= 64 | H100 only |
| `ligo_pwa` | Ligo AlphaFold3 Triton PWA kernel (Apache-2.0, patched: zero-init) | MSAPairWeightedAveraging | fwd + bwd | all |
| `megafold_tr` | MegaFold fused Transition (MIT) | Transition | fwd + bwd, bf16 | all |

Results: ext_attn_results.md (speed + accuracy per card). Reduced scope (coordinator): AF3 setup only (TriAttn d_pair 128 x 4 heads; AttentionPairBias 384 and the 768-channel x 16 heads token-DiT row; OPM / PWA / Transition AF3 rows), L 512 / 1024 / 2048, 20 iterations, 3 repeats. LayerNorm: no candidate has a standalone LayerNorm op (BioIR / MegaFold fuse it into LN+projection kernels), so no LayerNorm row.

Not connected: FlashBias (low-rank bias: not applicable to a dense pair bias; third_party/flashbias/README.md). `trifast` / `ds4sci` stay in `externals.py` (their pinned install is also recorded under third_party/).
