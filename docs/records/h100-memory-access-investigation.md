# H100 build 267166 memory-access investigation

Date: 2026-09-14. Local base: `fb6777417c0a34e383e0984978fec89169c18e28`.
Remote main reviewed: `d2266a035de11384c46f8cc980e6460f60925413` (eight newer commits).
Existing uncommitted H100 native-kernel work is preserved.

## Scope and confirmed failures

The user excludes capacity/OOM remediation from this investigation.

The two capacity errors are `augmented_attention`, dims 0, L=8192, training,
BF16 and FP32 compute. Both unit logs report `OutOfMemoryError` at line 23:
an allocation of 48.00 GiB with about 12 GiB free on an 80 GiB H100. Logs:

- `.scratch/h100-build/shards/logs/gpu2-augmented_attention-miniworld-bfloat16-corebfloat16-dims0-L8192-train-plan784b51229d6b-764954b8dd1d2ab1.log`
- `.scratch/h100-build/shards/logs/gpu0-augmented_attention-miniworld-float32-corefloat32-dims0-L8192-train-plan784b51229d6b-764954b8dd1d2ab1.log`

### Capacity recheck: actual shape and backward allocation

Rechecked after the user questioned the L=8192 attribution. Both shards record
`N_CTX=8192`, `A=48`, `B=1`, `H=4`, `HEAD_DIM=32` in the forward workload and
the same extents in backward preprocessing. Forward completed all 189 candidates;
backward preprocessing completed all 192. Thus the logged length is real, but
calling this simply an "L=8192 limit" omitted the augmentation and backward path.
The atom row of `registry_module.csv` declares train augmentation 48. Each unit
runs on one GPU; four build workers do not distribute a unit's activations.

CPU job 267385 called the actual unwrapped `_aa_bwd` launcher with meta tensors,
the same grid config directory, and no-op GPU launches. It recorded, in order:

| Allocation | Shape | Storage |
| --- | --- | ---: |
| `dq_expand` | `(256,48,1,8192,4,32)`, FP32 | 48 GiB |
| `dbias` | `(48,1,4,8192,8192)`, FP32 | 48 GiB |

The split count is `ceil(8192 / min(BLOCK_M2)) = ceil(8192 / 32) = 256`.
Both buffers coexist: **96 GiB before other live tensors**. This verifies the
allocation pressure without allocating physical GPU/CPU tensor storage. The
old log lacks an allocation traceback; its 58.49 GiB already allocated is
consistent with `dq_expand` having succeeded and the second 48 GiB allocation
failing. The exact failing allocation is an inference from that sequence, not
a recovered stack trace. No OOM behavior was changed.

### History and existing lower-memory alternative

The 2026-08-24 version at `401d4dae1` already has the same split-DQ and
augmentation-expanded DBias allocations. `docs/records/naming-audit.md` also
describes this memory cost and the atomic alternative. This establishes that the
design predates the H100 work, not that an earlier identical workload's OOM log
has been located.

CPU meta job 267387 verified the existing `_memeff_bwd` allocation sizes at the
same shape: DQ `(48,1,8192,4,32)` FP32 is 0.1875 GiB; DBias `(1,8192,4,8192)`
FP32 is 1 GiB. Atomic accumulation removes the split axis and accumulates bias
gradients across augmentations directly. These two buffers total 1.1875 GiB;
this is not a whole-operation peak-memory measurement (other tensors and the
returned bias layout conversion also consume memory). H100 performance and
numerical qualification remain pending.

Routing findings for a potential follow-up:

- The public kernel interface exposes `compute_efficient=False`, but the module
  invokes the interface with its default `True`.
- `whole_op.py`'s explicit `kernel_type="memory_efficient"` branch imports that
  public interface and calls it without `compute_efficient=False`; it therefore
  routes back to the default compute backend. The inspected remote has this same
  branch. No routing behavior was changed during this advice/history check.
- A buffer-size-based policy could choose the atomic path without hardcoding an
  L threshold. Other options are augmentation chunking or allocating split slots
  only after selecting a configuration; the latter alone leaves the 48 GiB DBias.

The device access failures are separate:

| Operation | Failures | Evidence |
| --- | ---: | --- |
| `augmented_attention_bwd_split_triton` | 6 | BF16 dims 2, HEAD_DIM=48, L=128 through 768; illegal memory access during sweep |
| `adaln_gemm_gate_triton` | 33 | BF16 direct-driver units; misaligned address during sweep |

The old logs did not print the fatal candidate. Therefore the candidates below
are inferred from the final finite measurement and grid order, not yet directly
confirmed failing launches:

| Operation | Last finite candidate | Next candidate to investigate |
| --- | --- | --- |
| attention backward | BLOCK_M1=32, BLOCK_M2=64, warps=2, stages=8 | Same tiles, warps=4, stages=1 |
| AdaLN GEMM | BLOCK_M=128, BLOCK_N=256, BLOCK_K=32, GROUP_M=1, warps=2, stages=4 | Same tiles/group, warps=4, stages=1 |

All 33 AdaLN fault shards have that same final finite configuration. For the
L=128 attention workload, Q/K/V/DO have shape `(48,1,128,16,48)` and strides
`(98304,98304,768,48,1)`. DQ has four independent split slots. The inspected
global-memory masks/strides do not establish an out-of-bounds access.

## CPU code generation evidence

`scripts/check-h100-memory-codegen.py` reconstructs the recorded ABI and compiles
eight candidates with the build's isolated Triton 3.6.0 environment for sm90:

| Operation | Warps | Stages | Shared bytes | WGMMA instructions in PTX |
| --- | ---: | ---: | ---: | ---: |
| attention | 2 | 1 | 33792 | 0 |
| attention | 2 | 2 | 37120 | 0 |
| attention | 4 | 1 | 28672 | 16 |
| attention | 4 | 2 | 41216 | 16 |
| AdaLN | 2 | 1 | 32768 | 0 |
| AdaLN | 2 | 2 | 40960 | 0 |
| AdaLN | 4 | 1 | 40960 | 8 |
| AdaLN | 4 | 2 | 81920 | 8 |

Artifacts: `.scratch/h100-memory-debug/codegen/` (PTX and TTGIR).
This supports investigating Hopper WGMMA lowering/scheduling. It does **not**
prove a compiler defect, identify the invalid instruction, or qualify a fix.
No grid candidates have been excluded based on this inference.

## Relevant remote changes integrated

The remote has no changes to either failing Triton kernel body. Its A5000 memory
investigations describe a different confirmed LNLinear schedule fault and other
faults that did not reproduce; those are not proof that these H100 faults are fixed.

The following remote capture/completion changes were adapted locally:

- Propagate a fatal CUDA error at warmup/synchronization immediately instead of
  retrying the same poisoned context through the benchmark callback.
- Print operation, configuration and cache key before propagating a fatal error.
- Write `_unit_complete` only when the unit ran successfully and recording succeeded.
- Resume/reclaim require a completed, compatible shard; partial timings alone
  cannot certify a unit. Keep partial measurements for later merge/reuse.
- Release failed-unit claims and stamp `.failed`, including a failed forced rebuild
  with an older completed shard still present.

The remote's unrelated model changes, A5000 cache publication and unit timeout
framework were not merged wholesale. Root `.git` was not changed.

## Validation and pending GPU work

CPU validation on `cpu-short`:

- Targeted error/completion/cache regressions: **82 passed**.
- Full `tests/autotune` and `tests/builder`: **1389 passed, 1 skipped**.
- Eight sm90 code generation comparisons completed.
- Ruff passed after formatting the new diagnostic scripts.

Diagnostic batch **267377** replaces still-pending 267349. Resources: H100 x4,
112 CPUs, 896 GiB, one hour, `gpu-4farm-bf`. The batch first runs eight isolated
Compute Sanitizer probes, then four focused original-unit sweeps. It does not
publish or merge caches. Output: `.scratch/h100-memory-debug/267377/` and
`.scratch/h100-memory-debug/slurm-267377.log`.

`scripts/repro-h100-memory.py` recreates original tensor shapes/strides/dtypes,
regenerates data, invokes one unwrapped JIT candidate, synchronizes, and compares
outputs/gradients to a torch reference. Each candidate runs in a fresh process.

At handoff the GPU batch is pending. **The device memory-access root cause and
kernel repair remain unverified and unfinished.** Once GPU evidence is available:
read the sanitizer's first failing instruction/address space, confirm the candidate,
repair that path, and rerun both the reproducer and relevant configuration grid.
