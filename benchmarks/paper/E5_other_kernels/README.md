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
