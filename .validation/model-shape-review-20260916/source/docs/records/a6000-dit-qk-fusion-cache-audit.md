# A6000 DiT, fused Q/K preprocessing and cache audit — 2026-09-11

AdaLN's repaired candidate set improves complete token DiT training. A new paired Q/K kernel
removes the strided-input copies and combines RMSNorm with 3D RoPE in SWA; its input backward
is fused as well. The existing sigmoid output gate remains fused.

## Controlled native module benchmarks

L384 (3072 atoms), depth 1, BF16 inputs/parameters, TF32 OFF, actual torch.compile ON.
Inference uses A5 and manual CUDA Graph replay; training uses A48, graphs disabled, and measures
forward+backward without an optimizer. These three modules have no dropout. Three timing samples
per implementation process, reported as the median. All implementations in each comparison ran
sequentially in one Slurm allocation on the same physical A6000. Compile evidence, input shapes,
GPU UUIDs, cache lookup evidence and samples are retained in the accompanying JSON.

| Module | Mode | PyTorch ms | MiniWorld before ms | MiniWorld after ms | Latency reduction |
|---|---|---:|---:|---:|---:|
| DiT (token) | inference | 1.0250 | 0.7393 | 0.7424 | -0.4% |
| DiT (token) | training | 18.0859 | 17.9313 | 17.5135 | 2.3% |
| SWA attention | inference | 0.3673 | 0.4080 | 0.3651 | 10.5% |
| SWA attention | training | 14.4224 | 10.1007 | 9.3338 | 7.6% |
| SWA-DiT (atom) | inference | 0.6241 | 0.5726 | 0.5283 | 7.7% |
| SWA-DiT (atom) | training | 21.5983 | 16.5406 | 15.7275 | 4.9% |

For token DiT, before substitutes only the saved pre-repair AdaLN forward-gate cache. For SWA
rows, both cases have the repaired AdaLN cache; before substitutes the old separate Q/K RMSNorm
and RoPE path. The final SWA run follows production-shape cache publication. Timing code and
module constructors come from the repository's native benchmark runner. Cache misses abort,
all timing processes assert tile-cache files remain unchanged, and final Q/K lookups additionally
require an exact stored physical-workload record. Profiling runs outside the timed samples.
Compilation is the repository's partial-graph mode; FA2 training's dynamic unpadding still has
Dynamo graph breaks. `compiled=true` is backed by observed compiled graphs, not a requested flag.

## Why the cache regression happened

The old system treated a logical bucket hit as proof that historical timing belonged to the
current workload. It merged raw milliseconds from different physical row counts and retained a
single top five. Small-input measurements could therefore discard the faster large-input
configuration. AdaLN's surviving one-warp candidates spilled registers; the runtime could retime
only the candidates that survived. Cache existence alone did not validate their admission.

The common Triton capture/write/reuse path now records tensor shapes, strides, dtypes, scalar
arguments and Triton's implementation dependency fingerprint, including called JIT helpers.
Each workload retains its own top five, and runtime reads the union of these candidates.
Shared round reuse, incremental searched sets, capture flush and shard merge use that identity.
A changed build workload clears Triton's in-process winner. An unprofiled old shard cannot
replace a profiled entry; profiled candidates remain authoritative if an older writer alters
only the flat compatibility view. Static coverage and cache-status checks also reject recorded
helper identities that runtime would reject. The policy applies to all 85 registered Triton ops.

Legacy entries remain usable as runtime candidate sets. Their unattributed times are not reused
as measured evidence for a new physical workload. This audit does **not** retroactively prove
that every legacy top five is optimal or that every historical shape was measured correctly.
A future build that actually visits an unattributed workload can require new measurements.

## Build input contract and cache wiring

The full derivation found another SWA mismatch: its build input factory supplied rotary tables
as `[N,S,1,half]` in activation dtype. Native SWA uses `[N,S,half]` FP32 tables. The builder now
uses that same shape and dtype, with a regression test for A5/A48. A failed derivation was not
published as a successful plan.

Actual L384 BF16 inference/training execution covered **22 module/mode cases**,
**57 distinct op/key pairs across 37 ops**.
Every selected tile belonged to its usable saved candidate set; all checked source identities
matched and no cache miss or heuristic fallback was accepted. External FlashAttention is outside
the MiniWorld autotune cache. This runtime audit covers the benchmark's 11 modules, not every
optional backend or arbitrary tensor shape.

The final source-stamped sm86 module plan has **1005/1005 usable keys**,
with no missing keys. This is declared-plan coverage on this A6000/toolchain, not certification
of another GPU. The sweep HTML was regenerated from the current plan.

New Q/K forward/backward ops are registered with drivers, numerical checks and 15-config grids.
Production-shaped cache builds cover A5/A48, FP32/BF16, four heads of width 32, and atom lengths
1024/2048/3072/4096/5120/6144/8192. Each physical workload searches the whole 15-config grid;
broadcast-table and ragged/partial-rotation checks are separate retained evidence.

## Numerical validation

The paired kernel is checked against explicit FP32 RMSNorm with a cast back to input dtype,
followed by the written PyTorch RoPE reference. FP32/BF16 output and input backward pass for
all 28 production dtype/shape combinations, plus broadcast/ragged cases. Tests also cover using
only Q in the loss, zero V gradients in interleaved QKV storage, non-power-of-two head width,
partial rotation and strided angle tables. Tolerances: FP32 relative L2 < 3e-6, BF16 < 8e-3. All 15 tile configurations are
also checked individually for both forward and backward at a ragged partial-rotation shape and
full A48/L384 in both dtypes: 60 config/shape/dtype cases, rather than only the selected winner.

Complete compiled SWA and SWA-DiT training at A48/L384 uses two nonzero random-weight seeds,
masked inputs including an empty sequence, and the same rounded weights in an independent
BF16 PyTorch implementation. Output, all applicable input gradients and all parameter gradients
pass; maximum gradient relative L2 is **1.128%** (acceptance bound 2.5%).
The reference shares the allowed FlashAttention backend but uses torch RMSNorm/RoPE/gating.

147 cache/registry/API/FA regression tests, 8 paired-Q/K GPU regression cases, and 18
build-input/coverage-policy regression tests passed (173 distinct cases). The final job reruns
the 26 build-input/coverage/QK cases after production cache publication. Scoped Ruff checks pass for the new kernel and changed cache/rope paths. See job
logs in `/home/psk6950/practice/miniworld-engine/.scratch/a6000-2026-09/mw-dit-qk-fix` for exact commands and test counts.

[Raw measurements, cache audit, plan coverage and numerical evidence](a6000-dit-qk-fusion-cache-audit.json).
