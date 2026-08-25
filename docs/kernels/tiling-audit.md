# Tiling audit

> **A record of one audit, not a current reference.** Kernel names here are the ones in
> `registry.csv` at the time it ran; where a kernel has since been renamed,
> `docs/kernels/rename-map.tsv` maps it. The *findings* — which axes are covered, which
> extents were never ragged — are what this file is for, and those still hold.

Every kernel the repo declares, measured against the tile configuration rather than the shape.
`docs/kernels/naming.md` covers what the kernels are called; this covers whether their tiling is
right. Nothing here is inferred from reading a kernel and deciding it looks correct — each row is
either a run on an A6000 (sm86) or an AST fact, and the two are labelled differently.

Harness, all under `src/miniworld_engine/tools/` and run from the repo root:
`tiling_sweep.py` (runs every registry driver, then its checker, under whatever
`MINIWORLD_CONFIG_DIR` names; `--isolate --dirty --repeat` are the three flags this audit learned
it needs), `tiling_static.py`, `grid_audit.py`, `mask_audit.py`, `gen_axes.py`, and the four
single-question probes. Results land in `.bench/`, which is scratch and untracked.

## Verdict

| | |
|---|---|
| kernels declared / driven / with a reference | 103 / **103** / **88** |
| tile configurations each kernel was run under | 9 config sets + 2 shape modes |
| **wrong numbers from a tile-size, axis-mixing, warp-count or GROUP_M change** | **0** |
| **masking bugs found (partial tile)** | **3 ops** — all 3 predicted by the static audit, all 3 measured, **all 3 fixed and re-verified** |
| **shared-memory race found (shape-independent)** | **1 kernel** — `layernorm_fwd_cuda`, **fixed and re-verified** |
| repeat-determinism, 4-8 runs per checker at one seed | every checker numerically in band; 1 kernel varies in bits, by design (fp atomics) |
| racecheck over every checker op | **0 hazards, 67/67** |
| alignment requirements, recorded with source evidence | 3 (+2 on kernels this card cannot run) |
| launch-only kernels: run at a partial tile, numbers never checked | **15**, of which 4 cannot run on this card at all |
| failures that are shared-memory capacity, not correctness | 19 (`blk128`) + 3 (`warp4`) + 5 (`warp8`) |

The three masking bugs, all on a GEMM contraction axis that no mask bounds:

| op | site | measured |
|---|---|---|
| `trimul_gemm_gate_mmajor_triton` | `bidirectional.py:88`, `:110` | wrong on a clean heap (2.9e-01), and 99.93% of outputs change when unrelated memory changes |
| `trimul_gemm_gate_saveact_triton` | `baseline_dtv1.py:280` (+ store `:350`, `:351`) | `ab=6.44e-01  sig_m=9.97e-01` vs 2.02e-03 aligned |
| `trimul_outproj_gemm_gate_saveact_triton` | `baseline_dtv1.py:400` (+ store `:435`, `:436`) | `ab=8.16e-01  sig=1.00e+00` vs 2.17e-03 aligned |

The static audit predicted exactly these three ops and no others; the runtime sweep failed exactly
these three and no others. Neither pass produced a false positive the other did not confirm.

### The fix, and what it was verified against

All three are the same defect: a K loop whose body never bounds the contraction axis. `K`, `N` and
the `BLOCK_*` tiles are all constexpr inside these kernels, so the bound can be dispatched at
compile time — the `EVEN_*` pattern `triangle_attention/triton/atomic.py:156` already uses. The
fully aligned path therefore keeps exactly the loads it had, and only a ragged extent pays for a
mask.

* `bidirectional.py` — the loop was `for _ in range(...)`, with no index to build a mask from, so it
  became `for k0 in ...` with `kmask = k0 + rk < K` ANDed into both loads under `if EVEN_K`.
* `baseline_dtv1.py` — `boundary_check=(0,)` bounded only axis 0 and the weight tiles had no
  `boundary_check` at all (unbounded on *both* axes); both loops now branch on `EVEN_K` / `EVEN_N`
  and check `(0, 1)` when either is ragged. The two unbounded store pairs became `(0, 1)`
  unconditionally — a store runs once per program, not once per K trip, and the sibling transposed
  branch already did this.

| kernel | aligned before -> after | ragged before -> after |
|---|---|---|
| `trimul_gemm_gate_mmajor_triton` | 2.97e-03 -> 1.91e-03 | **4.19e+00** -> **2.01e-03** |
| `trimul_gemm_gate_saveact_triton` | 2.02e-03 -> **2.02e-03** | **ab=6.44e-01 sig_m=9.97e-01** -> **ab=1.99e-03 sig_m=2.15e-06** |
| `trimul_outproj_gemm_gate_saveact_triton` | 2.17e-03 -> **2.17e-03** | **ab=8.16e-01 sig=1.00e+00** -> **ab=2.14e-03 sig=1.88e-06** |

The two `baseline_dtv1.py` kernels' aligned values are unchanged to every recorded digit, which is
the evidence that the compile-time dispatch really does leave the aligned path alone. The `sig`
error dropping to 1e-06 is the sigmoid no longer saturating on a contaminated accumulator.

Independent checks, both previously positive, both now clean:

```
memcheck  K=125 :  2229 errors  ->  0 errors      (K=128 control: 0 -> 0)
allocator clean vs dirty, K=125 :  929630 / 930250 elements differ  ->  0 / 930250 (bit-identical)
```

**Full registry regression after the fix** (`--isolate --seed 1234`, 103 kernels, both shape modes):
65 checkers pass in *both* modes, 0 wrong numbers. The only 6 failures are the cute kernels that
fail on architecture in aligned mode too (sm90/sm100 collectives on an sm86 card); one of them,
`trimul_gemm_gate_packed_sm100_cute`, fails earlier under ragged with the 16-byte pointer alignment
recorded as unknown below.

One self-inflicted detour worth recording: the first version of the `baseline_dtv1.py` fix passed
`boundary_check=(0,) if EVEN_K else (0, 1)`. Triton needs `boundary_check` to be a constant tuple
when the IR is built and does not fold a conditional expression in the argument, so both kernels
failed to compile (`CompilationError` at the `boundary_check` column). The fix has to branch at
statement level.

And a lesson about the audit tool rather than the kernels: the divisibility-guard rule that stops
the audit reporting the *fixed* code (a `(0,)` under `if EVEN_K:` is correct) could equally have
excused the original bug. That was checked, not argued — the two files were reconstructed with only
these edits reverted, and the audit still flags all 4 loops and all 4 stores on them.

## What the existing suite could not see

Three properties of the test matrix, each measured:

1. **Every driver extent is a multiple of 128** — `pair()` 128, `single()` 384, `rows2d()` 512x384,
   `drivers.layernorm_linear._M` 16384 — and every config set tiles at 16/32/64/128. So every extent divided
   every tile exactly and **no kernel's boundary mask had ever executed.**
2. **Every config set is axis-uniform.** All 91 ops in all 5 shipped sets gave every tile axis of
   an op the same value. A grid lambda dividing by the wrong axis name therefore computes the
   identical program count, and the mistake is arithmetically invisible.
3. **`num_warps=2` and `num_stages=1` in all 5 sets, all 91 ops, and in all 37 tuned cache
   entries.** Warp count and pipeline depth had never been varied by anything.

Of 210 tile axes across the 5 sets, 206 take 4 distinct values; the only exception is `GROUP_M`,
pinned at 8 in the 4 ops that have it.

## Config-set sweep (measured, A6000 sm86)

Each row is a full pass over the 91 triton kernels: run the driver, then the checker where one
exists, against a torch fp32 reference. `mixed1` / `mixed2` / `warp4` / `warp8` were added by this
audit to close gaps 2 and 3 above.

| set | kernels | numbers checked | launch-only | failed | failure cause |
|---|---|---|---|---|---|
| `accuracy` | 91 | 62 | 29 | 0 | — |
| `blk16` | 91 | 62 | 29 | 0 | — |
| `blk32` | 91 | 62 | 29 | 0 | — |
| `blk64` | 91 | 62 | 29 | 0 | — |
| `blk128` | 91 | 49 | 23 | 19 | 19 over shared-memory limit |
| `mixed1` | 91 | 62 | 29 | 0 | — |
| `mixed2` | 91 | 62 | 29 | 0 | — |
| `warp4` | 91 | 62 | 26 | 3 | 3 over shared-memory limit |
| `warp8` | 91 | 60 | 26 | 5 | 5 over shared-memory limit |

**Zero wrong numbers in every set.** Every failure is `OutOfResources: out of resource: shared
memory` at compile time, against the A6000's 101376 B limit:

* `blk128` — 19 of 91 ops cannot be built on this card at all (131072 B to 262144 B required).
  `configs/blk128` is therefore not a usable set on sm86 for those 19 ops, which nothing in the
  repo previously recorded. Grouped by what they need:

  | required | ops |
  |---|---|
  | 262144 B | `adaln_bwd_dlnw_triton`, `adaln_bwd_dw_triton`, `adaln_bwd_dx_dbias_triton` |
  | 196608 B | `cond_transition_fwd_b2b_triton` |
  | 131584 B | `cond_transition_fwd_b2b_saveact_triton`, `cond_transition_squeeze_gate_triton`, `cond_transition_squeeze_gate_saveact_triton` |
  | 131072 B | `cond_transition_bwd_gemm_triton`, `cond_transition_bwd_swiglu_dx_triton`, `cond_transition_bwd_swiglu_dx_packed_triton`, `cond_transition_expand_swiglu_triton`, `cond_transition_expand_swiglu_saveact_triton`, `layernorm_linear_fwd_fp32_triton`, `layernorm_linear_bwd_fp32_triton`, `trimul_gemm_gate_triton`, `trimul_bwd_gate_recompute_triton`, `trimul_outproj_gemm_gate_triton`, `trimul_outproj_bwd_gate_recompute_triton`, `trimul_outproj_layernorm_gemm_gate_triton` |

  The requirement is a static property of the kernel, so it can be compared against other cards'
  documented per-block maxima — sm86 101376 B, sm80 167936 B, sm90/sm100 232448 B. On that
  arithmetic (not measured; only sm86 was run) the three 262144 B adaln kernels do not fit on any
  of them, `cond_transition_fwd_b2b_triton` needs sm90 or newer, and the rest need sm80 or newer.

  `warp8` pushes the same three adaln ops to 163840 B and adds
  `cond_transition_bwd_swiglu_dx_triton` (163840 B) and
  `cond_transition_bwd_swiglu_dx_packed_triton` (147456 B); `warp4` costs those three adaln ops
  114688 B. All of those clear sm80's limit and fail only on sm86.
* `warp4` / `warp8` — `num_stages` 2 and 3 multiply the pipeline buffers; the same three adaln
  backward ops cross the limit at 114688 B. No kernel produced a wrong number at 4 or 8 warps.

`mixed1` and `mixed2` are the same tiles permuted in opposite order, so 73 of 91 ops have genuinely
unequal axes in both. **Both passed all 62 checks**, which is the only direct evidence available
that no grid lambda divides by the wrong axis — the uniform sets cannot distinguish that case.

`GROUP_M` swept over {1, 2, 3, 4, 16} for its 4 ops, including 3 (not a divisor of the 8 M-tiles)
and 16 (larger than the tile count). All four kernels produced **bit-identical output at every
value** and all were correct, which measures the claim at
`conditioned_transition/triton/train_fused.py:229` that GROUP_M is performance-only.

Reading the numbers: comparing `mixed1` against `mixed2`, 23 of 62 kernels write different bits
and 39 write identical bits. The identical ones are not evidence that the config was ignored — the
outputs are bf16, which rounds an accumulation-order difference away. The 23 that do change prove
the config reaches the kernels.

## Declaration cross-check (static)

`configs/<set>/<op>.csv` header vs the kernel's constexpr parameters vs `autotune/axes.csv`:

* **0** config columns the kernel does not accept (any such column is a launch error).
* **0** header differences between the 5 sets (a difference would change kernel arity per set).
* `axes.csv` was found naming axes that **do not exist in the code** in 88 of its 89 rows, with the
  axis *count* wrong for 19 ops too — e.g. `augmented_attention_bwd_atomic_triton` documented with
  `BLOCK_M`/`BLOCK_N`/`BLOCK_D` against an actual `BLOCK_M1`/`BLOCK_M2`. It had kept the normalised
  names from the rename that was reverted, so it could not be used to find a kernel's tile knob,
  and 3 ops had no row at all (`gated_projection_bwd_gate_recompute_flat_triton`,
  `trimul_gemm_gate_saveact_triton`, `trimul_outproj_gemm_gate_saveact_triton`, all in
  `modules/triangle_multiplication/baseline_dtv1.py`).

  **Regenerated** by `src/miniworld_engine/tools/gen_axes.py` from the code and the config CSVs: 210 axis rows over 91
  ops, every column a checkable fact — the op's kernel symbol and file:line, the axis name as
  triton injects it, its value in each of the 9 config sets, and `grid_axis` / `loop_step` /
  `loop_extent` / `extent_bounded`. The prose columns (`meaning`, `extent`, `notes`) were not
  re-derived, because re-deriving them is guessing; they are preserved verbatim at
  `docs/kernels/axes-legacy.csv`. The cross-check now reports 0 for all four defect classes, and
  the file surfaces the three unbounded axes below directly in the declaration.

## Grid audit (static)

96 autotuned kernel launches analysed. **0** launches divide by an axis the launched kernel does
not have. 7 launches sit in a function that reassigns one `grid` variable and uses it for kernels
of differing axis arity — `augmented_attention/triton/main.py:629,661`,
`memory_efficient.py:361`, `bias_only_attention/triton/main.py:426,441`,
`triangle_attention/triton/atomic.py:582,597`. That is the structure that made the
`BLOCK_M1`→`BLOCK_M` axis rename unsafe earlier: a rename tool cannot tell which assignment serves
which launch. `mixed1`/`mixed2` passing is the runtime evidence that all 7 are currently correct.

## Masking audit (static) — the findings

For each loop stepping an axis in `BLOCK` increments, does the loop body ever bound that axis?
Checking for the *presence* of `mask=` proves nothing: a row mask does not bound a contraction
axis, and an earlier version of this audit passed the kernels below because they are masked, just
not on the axis they walk.

133 tiled loops in 67 ops. **4 loops in 3 ops never bound the axis they walk:**

| op | site | loop | how it is unbounded |
|---|---|---|---|
| `trimul_gemm_gate_mmajor_triton` | `kernels/trimul_inproj/triton/bidirectional.py:88` | `range(0, K, BLOCK_K_D)` | `x` masked on rows, `w` masked on columns, nothing on `rk` |
| `trimul_gemm_gate_mmajor_triton` | `kernels/trimul_inproj/triton/bidirectional.py:110` | `range(0, K, BLOCK_K_D)` | same |
| `trimul_gemm_gate_saveact_triton` | `modules/triangle_multiplication/baseline_dtv1.py:280` | `range(0, K, BLOCK_K)` | 2 loads with no `mask` and no `boundary_check`; `xn` has `boundary_check=(0,)`, the M axis only |
| `trimul_outproj_gemm_gate_saveact_triton` | `modules/triangle_multiplication/baseline_dtv1.py:400` | `range(0, K, BLOCK_K)` | same |

These are latent bugs, not contracts. `bidirectional.py:135` asserts only `B == 1 and L == L2`;
`K` is `x_n.shape[-1]`, a free runtime dimension, and no launcher requires it to be a multiple of
anything. `BLOCK_K_D` comes from the autotuner, so the launcher could not assert against it even
if it wanted to. At the shipped `d_pair = 128` every tile in every set divides K exactly, which is
why this has never fired; `K = 192` with a tuned `BLOCK_K_D = 128` runs two trips over a 192-wide
row and multiplies 64 out-of-range columns into the accumulator.

Every `triangle_attention` loop passed: `main.py:72`, `:227`, `:316` load through block pointers
with `boundary_check=(0, 1)`, i.e. both axes bounded.

### Stores

The load audit above cannot see a write-side defect, and the transition family's run turned one up,
so a second pass covers every `tl.store` in every autotuned kernel. A store that writes `BLOCK_N`
columns into a row narrower than `BLOCK_N` spills into the *next* row — worse than an
out-of-range read.

Two corrections were needed before this pass produced signal rather than noise, both found by
checking its output against kernels whose ragged behaviour had already been measured:

* `mask` is `tl.store`'s third **positional** argument, so a keyword-only check misses
  `tl.store(p, v, m)`.
* A store inside `if EVEN_N and EVEN_D:` needs no mask — the kernel dispatches on divisibility and
  masks in the sibling branches. Without that rule the pass reported 5 false positives in
  `triangle_attention/triton/atomic.py:205,206,211,473,474`, whose ragged run measured clean. The
  rule has to be *divisibility* specifically, not "guarded by any compile-time flag": the broader
  version wrongly excused `baseline_dtv1.py:350`, which is guarded by a save-activations flag.
  Names qualify by being assigned from an expression containing `%`, or by being a constexpr
  parameter spelled `EVEN_*`.

Result: **5 stores correctly excused by a divisibility guard, and 4 unbounded stores — all in the
two `baseline_dtv1.py` ops already caught on the load side:**

| op | store | what is unbounded |
|---|---|---|
| `trimul_gemm_gate_saveact_triton` | `baseline_dtv1.py:350`, `:351` | `boundary_check=(0,)` writes `BLOCK_N` columns into an N-wide row |
| `trimul_outproj_gemm_gate_saveact_triton` | `baseline_dtv1.py:435`, `:436` | same |

`_input_gated_gemm_kernel` has **two** store branches and only one is affected: the transposed
(m-major) branch at `:331-332` uses `boundary_check=(0, 1)` and is safe, the row-major `else` branch
at `:350-351` uses `(0,)` and is not. Reading only the first branch is what makes this look fine.

Scope of this audit, stated precisely because a first version of it was narrower than it looked. It
covers every 3-argument `range` / `tl.static_range` loop in an autotuned kernel — 133 of them. An
earlier matcher required the callee to be spelled `range` and the step to be named `BLOCK_*`, which
silently dropped 6 real tiled loops: `trimul_inproj/triton/back.py:96,140`
(`tl.static_range(0, N, BLOCK_N)`) and the four `range(0, HEAD_DIM, HEAD_DIM_PAD)` loops in the
attention preprocess kernels. All 6 turned out to be properly bounded, so the finding set did not
change — but the narrower audit could not have known that. Still outside scope: index-based
persistent loops (`range(pid, num_tiles, G)` in `layernorm/triton/persistent.py:83` and
`layernorm_linear/triton/mmajor_bwd.py:80`, `range(num_splits)` in
`augmented_attention/triton/main.py:541`) and `tl.static_range(NH)` head loops, where the range
itself bounds the index rather than an extent being tiled.

## Ragged extents

`MINIWORLD_SHAPE_MODE=ragged` (`kernels/drivers.py`) subtracts 3 from each extent routed through
`ragged()`, making the tail tile partial for all tile widths at once; an extent a kernel's own
contract forbids varying goes through `aligned_only(label, n, why)`, which keeps it aligned and
records the reason so the sweep never counts it as ragged coverage it did not have.

All five families ran. Two findings need their own explanation before the per-family tables:

* **`trimul_gemm_gate_mmajor_triton` reads out of bounds at a ragged K, and whether that shows up
  in the numbers depends on what is in the memory it reads.** At `K=125` with `BLOCK_K_D=64` the
  loop runs two trips and the second reads `rk` 64..127, so columns 125..127 fall off the row: the
  accumulator picks up the *next* row's first three channels, and past the last row it reads beyond
  the tensor, plus three rows past the end of the `(125, 4*H2)` weight.

  Measured both ways, and the two disagree:

  | run | rel (left) |
  |---|---|
  | ragged, single process | 4.19e+00 |
  | ragged, `--isolate` (fresh subprocess) | 2.01e-03 |
  | aligned `K=128` | 3.43e-03 |

  The contaminating term is `x_extra * w_extra`. `x_extra` is the next row's channels and is always
  non-zero, but `w_extra` lies past the end of the weight, and in a fresh process that memory is
  still zero — so the garbage products are zero and the bug hides. The same failure mode is already
  documented for a different kernel at
  `tm1/cute/sm100_gate_gemm_collective.py`: "once the caching allocator has served non-zero memory,
  leaks NaN."

  **Consequence for the harness.** `--isolate` is required to avoid the cascade described below,
  but it zeroes fresh memory and therefore *masks* allocator-dependent out-of-bounds reads — here it
  produced a false negative. Neither mode alone is sufficient.

  **Settled by a controlled experiment** (`src/miniworld_engine/tools/trimul_bidir_oob.py`), which holds shape,
  config and seed fixed in one process and varies only whether the caching allocator has served
  non-zero bytes:

  ```
  config {'BLOCK_K_D': 64, 'BLOCK_K_H2': 64, 'BLOCK_M1': 64}
  K=128  clean heap  rel_left=1.968e-03
  K=128  dirty heap  rel_left=1.968e-03    bit-identical      0 / 952576 elements differ
  K=125  clean heap  rel_left=2.914e-01
  K=125  dirty heap  rel_left=1.259e+01    NOT bit-identical  929630 / 930250 differ
  ```

  **Confirmed by `compute-sanitizer --tool memcheck`** under
  `PYTORCH_NO_CUDA_MEMORY_CACHING=1`, on a minimal single-launch repro:

  ```
  memcheck  K=128 :     0 errors        (control)
  memcheck  K=125 :  2229 errors
     Invalid __global__ read of size 8 bytes
         at _bidir_front_kernel+0x31b0 in bidirectional.py:90
         and is 2,001 bytes after the nearest allocation of size 250,000 bytes
             Host Frame: bidir_front_triton in bidirectional.py:148
     all 100 printed errors at :90; offsets 2,001 -> 4,241 bytes past the allocation
  ```

  250,000 bytes is exactly the `(125, 4*H2 = 1000)` bf16 weight; `bidirectional.py:90` is the
  `tl.load(w_ptrs, ...)` inside the K-loop; `:148` is the launch. So the defect is established three
  independent ways — the static absence of a bound on `rk`, the dependence on memory the kernel
  does not own, and the sanitizer naming the instruction, the access size and the offset.

  Two further limits of the tooling, worth keeping because both look like exonerating evidence and
  are not:

  * **Only the weight load is flagged, not the `x` load at `:89`** — and that is consistent rather
    than contradictory. The `x` overrun is 3 elements (6 bytes) past a 930,250-byte tensor and hides
    inside `cudaMalloc`'s own padding; the weight overrun reaches ~6,000 bytes and clears it. So
    even with caching disabled, memcheck under-reports small overruns.
  * **`--tool initcheck` returned 0 errors at K=125.** That is its scope, not a clean bill: initcheck
    flags reads of uninitialised memory *inside* a valid allocation, and an access wholly outside
    one is memcheck's domain. Reaching for initcheck here (as this audit did) was the wrong
    instrument.

  **On the evidence's direction.** This experiment is asymmetric: a clean result would prove
  nothing (it is consistent with reading memory that happened to be zero), but the observed
  *dependence* is one-directional proof that the kernel reads outside its operands. What it does
  not give is the instruction, the access size, or the offset relative to the allocation.

  **And on the tool that would give those: `compute-sanitizer --tool memcheck` has a blind spot
  here.** It tracks *driver* allocations, while PyTorch's caching allocator makes a few large
  `cudaMalloc`s and sub-allocates out of them. This overrun is 3 x 4*H2 = 3000 elements (6000
  bytes) past the end of a 250,000-byte weight — comfortably inside the same driver allocation, so
  there is no boundary to cross and memcheck reports `0 errors` while the kernel is reading out of
  bounds. A memcheck run on a torch workload is only meaningful with
  `PYTORCH_NO_CUDA_MEMORY_CACHING=1`, and even then a fresh heap returns zeros and hides the wrong
  number, so the run has to be read for sanitizer output rather than for `rel`.
  `--tool initcheck` is the more direct instrument for this claim: it flags reads of device memory
  that was never written, which does not depend on crossing an allocation boundary at all. (This
  is also why the earlier `atomic.py` out-of-bounds read *was* caught by memcheck — that one
  crossed a real allocation boundary. The tool is not unreliable; its unit of tracking is not
  torch's.)

  `K=128` is bit-identical however the heap is dirtied. `K=125` is already wrong on a clean heap,
  two orders outside the bf16 band, and **99.93% of its output elements change when unrelated
  memory changes**. A kernel that reads only its own operands cannot do that. This is stronger
  evidence than a sweep in either mode, and stronger than a wrong number on its own: it
  demonstrates the dependence on memory the kernel does not own.
* `layernorm_bwd_split_cuda` raises `CUDA error: misaligned address` at a 125-wide row. Under
  investigation as an alignment requirement of its vectorised loads rather than a masking bug.
* **A CUDA error poisons the context.** After the first illegal or misaligned access, every later
  launch in the same process re-raises that same error. The first ragged run reported 16 failures
  with byte-identical error text, of which one was real and 15 were never measured. Use
  `src/miniworld_engine/tools/tiling_sweep.py --isolate`, which runs each kernel in its own subprocess. Any sweep that
  can hit a memory fault and does not isolate is reporting the first failure N times.

## Ragged results by family

`rel` from the isolated, seeded pass unless noted. "launch-only" means the kernel has no checker,
so a ragged run proves it does not fault and says nothing about its numbers.

### trimul (25 kernels, 15 with checkers)

Extents perturbed: `D = ragged(128)` -> 125 (channel width, simultaneously the LN reduce axis and
the GEMM contraction `K`, carrying `2D`/`4D`/`5H`/`h2` and the `(D,)` LN weights with it) and
`L = ragged(64)` -> 61, which makes *both* spatial axes of the `[B, L, L, D]` pair ragged at once
and carries `M = L*L` 4096 -> 3721.

* **14 mask correctly** — all in the 1.8-3.8e-03 bf16 band at both shapes.
* **1 masking bug** — `trimul_gemm_gate_mmajor_triton`, above.
* **6 launch-only** — `trimul_gemm_gate_triton`, `trimul_outproj_gemm_gate_triton`,
  `gated_projection_bwd_gate_dropres_triton`, `trimul_bwd_gate_packed_triton`,
  `trimul_gemm_gate_packed_mmajor_triton`, `trimul_outproj_gemm_sigmoid_triton`. No checker, so a
  mask bug in them would not be caught numerically. `front.py::_lr_kernel` reads as correctly
  masked on `rk`, but that is a source read, not a measurement.
* **4 cute kernels fail on architecture in *aligned* mode** (`expects arch to be sm_90a/sm_100a,
  got sm_86`), so none of their ragged behaviour is a tiling verdict.

Alignment requirements recorded with evidence rather than assumed:
`persistent_dense_gemm_kernel`'s `n`/`k` (`_blackwell_dense_gemm.py:1287` requires `% 8 == 0` for
bf16, raises at `:1295`), and `tm2_dual_kernel`'s `K`/`M` (`tm2_cute_kernel.py:66`
`assert K % _TILE_K == 0`, `:406` `assert M % tile_m == 0`).

Left as unknown rather than guessed:

* `trimul_gemm_gate_packed_sm100_cute` fails *earlier* at ragged with `Tensor data pointer is not
  aligned to 16 bytes`, from the launcher's own `from_dlpack(..., assumed_align=16)` on
  `preact[1::2]` whose base is offset `2M` bytes, needing `M % 8 == 0`. Genuine TMA requirement or
  unhandled case is not decidable without an sm100 device, so it was left ragged and visible.
* `trimul_gemm_gate_sm100_cute` pads `M` up to a 128-multiple inside `gate_gemm`, absorbing the
  ragged `M` before the kernel sees it. Its masking is untested, not proven.

### layernorm / layernorm_linear / fused_ln_mask (18 kernels, 10 with checkers)

Extents perturbed: `_M` 16384 -> 16381 (the LN row axis), `_D` 128 -> 125 (the LN **reduce** axis,
carrying every `vec(_D)` gamma/beta and the `(N, K)` projection weight with it), `_PAIR_N`
128 -> 125 (pair seq length, so the flattened `B*L*L` row count becomes 15625), `_NH` hoisted from
an inlined `_D//32` so driver and checker cannot drift apart.

* **15 mask correctly** — 8 of those are launch-only, i.e. proven to run at a partial tile, not
  proven numerically right.
* **0 masking bugs.**
* **2 alignment required, with evidence** — `layernorm_bwd_split_cuda` and
  `layernorm_bwd_reduce_cuda` (same launcher) raise `CUDA error: misaligned address` at `_D=125`.
  `layernorm/cuda/layer_norm_cuda_kernel.cu:276` loads
  `*reinterpret_cast<const VecT*>(x + col0)` with `VecT` = uint2/uint4; `:269` states the
  precondition "N % EPT == 0 for every launched (N, TX_BYTES) combo"; `:465` chooses only between
  two vector widths (`txb = (N % (32*(16/elt)) == 0) ? 16 : 8`) with **no scalar tail**. For bf16,
  EPT = 4, so N = 125 puts the row base at 250 B — 2 mod 8 on odd rows — and overruns the row by 3.

  Only the *feature width* is pinned: `_M` stays ragged for these two and passes (2.18e-03 /
  3.13e-03), and the scalar forward kernel keeps a plain `_D` and passes at 125. That is what
  distinguishes "the vector path requires alignment" from "CUDA fails at 125".

**A robustness defect found alongside, not a tiling one:** that `N % EPT` precondition is nowhere
enforced — no `TORCH_CHECK`, no scalar fallback. An out-of-contract `N` faults instead of being
rejected. The `aligned_only` wrap documents it; it does not fix it.

**Unknown:** `layernorm_linear_fwd_sm90_cute` fails in *both* modes with
`AssertionError: SM90 (H100) only`, so its ragged behaviour was never measured. Its
`_reduce_gmem_coop` docstring says "Requires len_k % 32 == 0", which suggests it would need a
pinned K too, but that was not verified and `_D` was left ragged so the failure stays visible on
Hopper.

### attention: triangle / augmented / bias-only (17 kernels, 13 with checkers)

Extents perturbed: `L` 128 -> 125 (sequence length — bounds the query loop *and* the key/value
loop in all 6 kernel files, and carries `M = L*L` and the flattened `HL = H*L` program axis),
`D` 32 -> 29 (head dim), `DH` 128 -> 125 (gate-out contraction) and `DP` 128 -> 123 (gate-out
output width). `DP` was perturbed by 5 rather than 3 on purpose: `_dgrad_epi` tiles `N` as its
contraction and `DH` as a free axis, and while the two are equal a swapped mask still looks right.

* **13 checked kernels, all mask correctly. 0 masking bugs**, across 5 sweeps
  (aligned/ragged x seed 1234, aligned/ragged x seed 4321, aligned unseeded). The `sha` column
  differs between aligned and ragged for all 13, so the perturbation demonstrably reached every
  kernel instead of being folded away.
* This corroborates the static audit: `main.py:72`, `:227`, `:316` all come back correct at a
  ragged seqlen. `augmented_attention_bwd_reduce_triton` ran at `N_ELEM = 116000`, a genuinely
  partial `BLOCK_E` tail, and matched exactly.
* **1 alignment required, with two independent pieces of evidence** —
  `triangle_attention/triton/atomic.py:518-519` raises
  `ValueError(f"Only support D=32, but got {D=}")`, and separately at `:57-66` the `not EVEN_N`
  branch of `_attn_fwd_inner` drops the `offset_k < HEAD_DIM` mask entirely, which is only safe
  because `EVEN_D` holds at `D == HEAD_DIM_PAD == 32`. Only the head dim is pinned; that file's
  seqlen stays ragged and passes.
* **`HEAD_DIM` was NOT treated as an alignment requirement for the other three files**, and that
  restraint was checked rather than assumed: `HEAD_DIM_PAD = next_power_of_2(HEAD_DIM)` with
  `< HEAD_DIM` masks and `boundary_check=(0,1)` is the mechanism that *supports* a non-power-of-two
  head dim, and `D = 29` passes on all three.
* **`H` (4) and `A` (8) were not perturbed** — they appear only in grid extents and stride
  arithmetic, never blocked by a `tl.arange`, so there is no boundary mask to exercise. That is an
  inspection claim, not something the sweep tested.

Unknowns recorded rather than smoothed over: the three `bwd_pre` checkers return *exactly*
`0.00e+00` in ragged mode at both seeds (against ~7e-08 aligned); the plausible reading is that the
masked tail contributes exact fp32 zeros so triton's 32-lane reduction tree lands on torch's value,
but the two trees were not proven to coincide. And the four launch-only forwards have no reference,
so "ok" means only "did not raise" — each is indirectly constrained by the backward checker that
consumes its `out`/`m`, which is weaker than a direct forward reference.

### adaln / conditioned_transition (26 kernels, 15 with checkers)

Four constants became four independently perturbed axes: `_M` 512 -> 509 (row axis), `_D`
128 -> 125 (`d_hidden` = `NX`), `_DC` 128 -> **123** (`d_cond` = `NC`, split out of `_D`, which
previously served both), `_ND` 512 -> 509 (expand width, perturbed on its own axis rather than
inherited as `4*_D`).

Two of those choices are the reason the family is actually covered:

* **The two hidden widths are ragged to *different* values.** `_DC` uses `by=5` where `_D` uses
  `by=3`, so `d_cond != d_hidden`. While they were equal, a mask bug on the `NC`/`DC` axis — or a
  launcher reading one width where it means the other — stayed invisible. Both remainders were
  checked partial for all of 16/32/64/128.
* **`_ND` is not derived from `_D`.** `4*125 = 500` would also be partial, but the tail launchers
  take `ND` from `wa.shape[0]` and `D` from `ws.shape[0]` with no relation between them, so an
  independent 509 keeps ND's remainder from coinciding. The derived `2*_ND = 1018` with
  `BLOCK_K = 64` additionally makes `_dx_swiglubwd_kernel`'s `j` tiles straddle the `ND` split
  point, exercising its `n = j % ND` / `is_db = j >= ND` logic.

Results: **15 checked kernels all mask correctly, 0 masking bugs, and `aligned_only` was used
nowhere** — `grep` for `assert` across all 8 kernel files finds only `set_forward_mode` /
`set_wgrad_backend` string asserts, so no kernel in either family has a shape contract to honour.
11 kernels are launch-only and therefore unverified numerically at ragged shapes, including all
four `adaln/triton/main.py` kernels: their `NX`/`NC` masks are now exercised, but only against "did
not fault".

Aligned mode came back **identical to `.bench/tiling/accuracy.csv` on all 26 rows** — phase, ok and
every recorded digit — and was reproduced by a second independent pass. (An exact match is
obtainable when a kernel's rel is dominated by quantisation rather than by the draw; it was not
obtainable for the other families, which is the baseline defect recorded below, not a contradiction.)

**GROUP_M, examined more closely than the sweep could.** `_gemm_gate_kernel` (and `_dgemm`,
`_dh_gatebwd`, `_dx_swiglubwd`) compute `pid_m = first_m + (pid % group_size)` rather than the
canonical `((pid % width) % group_size)`. Ragged-by-3 does not reach the partial-group path at all:
`cdiv(509, 64) == cdiv(512, 64) == 8 == GROUP_M`, and the tile count is unchanged at the other tile
widths too. So it was settled two ways — arithmetically (within the last group the
`group_size` consecutive `r` hit every residue exactly once, so the mapping is a permutation of the
canonical one, not a loss of coverage), and with a dedicated probe at `M = 573`
(`grid_m = 9`, `GROUP_M = 8`, `grid_n = 2`, i.e. a last group of size 1), which came back in band
across three runs. Together with the GROUP_M sweep above — which varied the group at a fixed 8
tiles, including a non-divisor 3 and an oversize 16 — the swizzle is covered from both directions.

### transition / triangle_multiplication (14 kernels, 14 with checkers)

Extents perturbed: `ROWS` 4096 -> 4093, `K_SMALL` 128 -> 125 and `K_LARGE` 256 -> 253 (the LN
feature width, which is simultaneously the GEMM contraction extent, dragging `wa`/`wb`/`ws` and
`gamma`/`beta` with it), `ND_SMALL` 512 -> 500 (expand/gate output width), `TRIMUL_ROWS`
16384 -> 16381, `TRIMUL_D` 128 -> 125. `N_EXPAND` stays 4 — it is the op's expansion factor, not a
tile extent. Every tile in `configs/accuracy` is 64, so all seven extents get a partial tail.

* **11 mask correctly.**
* **2 MASKING BUGS, measured** — the two `baseline_dtv1.py` ops the static audit predicted:

  | op | aligned | ragged |
  |---|---|---|
  | `trimul_gemm_gate_saveact_triton` | 2.02e-03 | `ab=6.44e-01  sig_m=9.97e-01` |
  | `trimul_outproj_gemm_gate_saveact_triton` | 2.17e-03 | `ab=8.16e-01  sig=1.00e+00` |

  Numeric failures in independent subprocesses, so unlike the `bidirectional.py` case these do not
  depend on allocator state — the weight loads carry no `boundary_check` at all, so the read lands
  further out. The K-axis defect alone accounts for `rel ~ 1`; the N-axis store defect above is a
  second, independent contribution that was not separated experimentally.
* **1 untested** — `transition_bwd_swiglu_gate_sm100_cute` fails identically in both modes
  (`expects arch to be one of [Arch.sm_100a, ...]`); the partition is sm86 only.
* **`aligned_only` used nowhere, and the reason was checked rather than assumed.** The hand-CUDA
  fixed-shape contract at `kernels/transition/cuda/__init__.py:116-118` and the sm90 WGMMA
  `K in {128,256}` gates at `triton/fused.py:1105-1119`, `:1325-1329` are real, but **none of these
  14 kernels reaches either** — all 11 driven transition kernels call the Triton or cutlass-DSL
  launcher directly (`layernorm_bwd_privatized_triton` explicitly sets
  `transition_lnbwd_cuda=False` to stay off the CUDA route), and the three hand-CUDA registry rows
  from the other `.cu` file have no driver and report `no-driver` in both modes.

Cross-family effects this run surfaced: `layernorm_bwd_split_cuda` / `layernorm_bwd_reduce_cuda`
faulting at a ragged feature width (the alignment requirement recorded under layernorm above), and
`trimul_outproj_gemm_gate_sm90_cute` failing earlier with
`AssertionError: M=3721 must be divisible by tile_m=64` — an explicit contract.

## The race that the tiling sweep was not looking for

Verifying the three masking fixes turned up a fourth defect of a different kind, and it is the one
worth carrying forward, because the whole verification apparatus up to that point could not see it.

`layernorm_fwd_cuda` gave `rel rstd=7.05e-01` in one run under compute-sanitizer and passed
everywhere else. 20 controlled repeats at the identical shape outside the sanitizer were
bit-identical and correct, so the difference was the instrumentation — which perturbs timing.
`compute-sanitizer --tool racecheck` then named it outright:

```
Race between Read  @ layer_norm_fwd_kernel<BFloat16>+0x3f0
         and Write @ layer_norm_fwd_kernel<BFloat16>+0x5b0   [~70000 hazards]
aligned  d=128   rel rstd=9.69e-01   y=1.24e+00
ragged   d=125   rel rstd=8.56e-01   y=8.65e-01
```

**Both shapes.** This was never a tiling bug — it is a pre-existing race at the shipped default
width, and the kernel was inside this project's earlier "94/94 numbers verified" result. That
result was not wrong about what it measured; it measured each kernel *once*, and a race that
manifests on a fraction of runs cannot be seen that way.

`layer_norm_cuda_kernel.cu` had a `block_reduce_sum` with no trailing `__syncthreads()`, and its
caller broadcasts through the same buffer:

```cpp
sum = block_reduce_sum(sum, smem);
if (threadIdx.x == 0) { mean = sum / (float)N; smem[0] = mean; }
__syncthreads();
mean = smem[0];              // every warp reads smem[0] here
var = block_reduce_sum(var, smem);   // warp 0 lane 0 writes smem[0] again
```

Passing a barrier is not the same as having issued the load. Warp 0 can finish the pass-2 loop and
write its partial into `smem[0]` while a slower warp has not yet read `mean` — that warp then uses
a partial sum as the mean and poisons the variance. memcheck is silent because nothing goes out of
bounds; the forward kernel reads `x[i]` scalar under `i < N`. Fixed with two barriers: one at the
end of `block_reduce_sum`, making the helper safe to call twice on one buffer, and one after the
broadcast read. The same pattern exists nowhere else in the repo's CUDA sources.

| check | before | after |
|---|---|---|
| racecheck, aligned | ~70000 hazards, rstd 9.69e-01 | **0 hazards**, rstd 8.97e-08 |
| racecheck, ragged | ~70000 hazards, rstd 8.56e-01 | **0 hazards**, rstd 1.77e-07 |
| 5 repeats inside memcheck | 3 distinct output hashes | **1 hash, bit-identical** |

### Closing the axis it exposed

Two sweeps, because one kernel behaving this way says nothing about the other 66:

* **racecheck over every checker op** (67, sharded): **0 hazards everywhere.**
* **repeat-determinism**, 4 runs per checker at one seed, both shape modes, dirty allocator: every
  checker numerically in band. One kernel varies in *bits* —
  `augmented_attention_bwd_atomic_triton`, 7 of 8 repeats distinct — and that is correct behaviour,
  not a defect: floating-point `atomic_add` reorders its accumulation every run. Corroborated by
  racecheck reporting 0 hazards for it and all 8 repeats staying in the 1.9-4.9e-03 band.

The first version of that test called any bit difference `NONDETERMINISTIC: reads memory it does
not own` and overwrote the rel value, which mislabelled exactly that kernel. The rule now fails
only on a repeat that is numerically wrong, and reports the hash spread beside the rel.

## A methodological defect in the aligned baseline

`.bench/tiling/accuracy.csv`, used as the "did anything change" baseline, was recorded **without a
seed**, and the checkers draw fresh `randn` on every call. So an exact match to it was never
obtainable. Measured rather than assumed: two unseeded aligned passes at byte-identical shapes
disagree by more than any aligned-vs-baseline gap — `gated_projection_gate_packed_mmajor_triton`
3.79e-03 vs 2.06e-03 (1.8x), `gated_projection_bwd_gate_triton` 2.49e-03 vs 3.24e-03. Shape
identity has to be established from the code (`ragged()` returns `n` unchanged outside ragged mode,
and the module reports `D,L,M = 128 64 4096`) and from phase/ok being unchanged, not from the rel
value. Any future baseline should be recorded with `--seed`.

The layernorm family quantified the same thing independently and put a number on it: the maximum
spread between two identical unseeded runs was **7.40e-04**, while the maximum
|aligned - `accuracy.csv`| gap was **6.30e-04** — i.e. the whole discrepancy sits *below* the RNG
floor. Its seeded aligned run also reproduced bit-identically across two different Slurm jobs on
all 18 rows, so the harness itself is deterministic once seeded; only the baseline was not.

## Out-of-bounds sweep (memcheck, per op)

A rel value cannot answer "does this kernel read outside its operands"; the trimul bug passed the
isolated ragged sweep while reading 2 KB past a weight. So every op with a checker was re-run at
ragged shapes under `compute-sanitizer --tool memcheck`, with `PYTORCH_NO_CUDA_MEMORY_CACHING=1`.

**This has to be done one sanitizer process per op, and NOT with `--isolate`.** Measured: with
`--isolate --target-processes all`, a 2-op run still emits exactly **1** `ERROR SUMMARY`, and in
isolate mode the parent launches no kernels at all — so a whole-registry run that way instruments
a process that does nothing and reports a meaningless `0 errors`. That first attempt was discarded
for this reason. (A second self-inflicted one: the per-op job omitted `--backends`, whose default
is `triton`, so the 5 non-triton checker ops silently ran zero kernels and the sanitizer said
"Target application terminated before first instrumented API call". `NO_SUMMARY` is not `0 errors`.)

| | |
|---|---|
| triton ops with a checker | **62 — `ERROR SUMMARY: 0 errors`, 0 `Invalid` lines** |
| `layernorm_bwd_split_cuda`, `layernorm_bwd_reduce_cuda` | 0 errors, numbers correct |
| `layernorm_fwd_cuda` | 0 errors; one unreproduced wrong-number observation, below |
| 2 cute ops with a checker | not measurable — they never launch on sm86 |

### The one unreproduced observation

In a single run under memcheck, `layernorm_fwd_cuda` returned `rel mean=4.36e-08 rstd=7.05e-01
y=4.46e-01` while memcheck itself reported 0 errors; the same command in the same job had passed
moments earlier. It has **not** reproduced in 27 subsequent runs: 20 repeats of the registered
checker (ragged and aligned x caching on/off x dirtied allocator, all bit-identical, `rstd`
1.773e-07), and 7 repeats of the exact failing command — 3 plain, 3 under memcheck, 1 under
memcheck with `CUDA_LAUNCH_BLOCKING=1` — all passing.

It is recorded rather than dismissed because the shape of the error is not noise: `mean` was
correct to every digit while `rstd` and `y` were wrong, which is partial corruption, not a bad
draw. Cause undetermined. Note also that a small overrun here would be invisible to memcheck
anyway — a 125-element bf16 row is 250 bytes and a 3-element overrun 6 bytes, inside `cudaMalloc`'s
own alignment padding, the same blind spot that hid `bidirectional.py`'s `x` load.

One earlier version of the probe for this hardcoded `rows2d(4096, d)` and found the kernel
deterministic — at a row count the driver never uses (`_M` is 16381). It proved nothing until it
called the registered checker instead.

## Closing the coverage gap the audit measured

The audit's own biggest finding was not a kernel: 33 kernels had a driver and no checker, and 3 had
no driver at all. A driver proves a kernel *runs*; a wrong answer with no exception is recorded as
a pass. That is precisely the state the three masking bugs above were found in — they were caught
only because they happened to have checkers.

| | at session start | after the tiling audit | now |
|---|---|---|---|
| with a reference | 35 | 67 | **88** |
| driver only, numbers unverified | 56 | 33 | **15** |
| no driver at all | 3 | 3 | **0** |

21 checkers were added: 8 layernorm / layernorm_linear / fused_ln_mask, 4 attention forwards, 6
trimul, 3 for the vendored `transition_cuda` extension. All pass in both shape modes under
`--isolate --dirty --repeat 3`. Each was audited before being wired, because a checker that
compares a kernel to itself passes just as green as a real one; three worth naming:

* the attention forwards identify the saved `m` as a **base-2 log-sum-exp**, not the running row
  max — checking it against the max would be wrong by `log2(l)` on every row and would still look
  plausible, which is exactly why the indirect constraint through the backward was weak;
* `layernorm_stats_triton` compares against the *family's* definition of the statistics
  (`_ln_stats`, biased variance) rather than the kernel's route to them (tiled `E[x^2] - mean^2`),
  so an algebra error there is exposed instead of reproduced;
* `trimul_gemm_gate_packed_mmajor_triton` builds its reference from the **unpacked** WL/WLg/WR/WRg,
  so it cannot agree with the kernel about the interleaved `(D, 4D)` packing by construction.

That last one settles an item this audit had left as a source-reading claim:
`front.py::_lr_kernel` has the same loop shape as `_bidir_front_kernel`, which was found reading
2001-4241 bytes past its weight at K=125. It was *read* as correctly masking `rk`; it is now
**measured** clean at K=125 (rel 2.08e-03).

### One elevated number, measured rather than argued

`adaln_fwd_saveact_triton`'s saved `Gate` comes back at 1.5-2.8e-02 against the fp32 reference,
14x the other pairs in that checker (`Y` is 5.7e-03, `XHat`/`CondNorm`/`Rstd*` ~1e-07) though well
inside the 5e-02 band. `main.adaln_fwd_kernel` computes `cond_aff = cond_norm * lnw` in fp32
registers and casts it to bf16 before the two `tl.dot`s, so `scale` carries the cast's error before
the sigmoid sees it. That was the reasoning; this is the measurement — the same reference with a
bf16 cast at that one position and fp32 everywhere else:

```
                              aligned s1234  aligned s4321  ragged s1234  ragged s4321
Gate vs fp32 reference          2.306e-02      1.708e-02     1.755e-02     1.517e-02
Gate vs bf16-operand reference  2.397e-03      1.953e-03     1.953e-03     1.953e-03
the two references differ by    2.387e-02      1.644e-02     1.858e-02     1.562e-02
```

The error collapses into the band every other pair sits in, and the gap between the two references
accounts for essentially all of the discrepancy. `1.953e-03` is 2^-9 — bf16 unit roundoff exactly.
`scale` has std ~9 (the predicted sqrt(NC) order) and `max|Gate|` is 1.0, i.e. saturated.

So the kernel is as accurate as its own operand precision permits, and what remains is the
checker's fp32 reference being stricter than the kernel — which is the correct behaviour for a
checker, not a defect in either. Reproduced across two seeds and both shape modes.

The 4 that remain: cute kernels that cannot run on sm86 at all. Previously also the adaln family's 11,
now written and wired (11/11 passing in both modes). Two hazards made that family the delicate
one, and both are recorded because getting either wrong produces a checker that passes and means
nothing:

* its drivers pass `mean`/`rstd` as `ones` — fine to *reach* a kernel, but outside its valid
  regime, where agreement means nothing. The checkers supply real statistics at the same shapes.
* a LayerNorm *backward* reference must treat mean/rstd as functions of x, or it silently omits the
  two centering terms. Saved *activations* (x_hat, cond_norm, gate) are consumed as-is instead, so
  the reference reads the same numbers the backward kernels read. The trick that gets both at once:
  `F.layer_norm(x_hat / rstd)` reproduces the saved `x_hat` exactly — LayerNorm is shift-invariant
  and `var(x_hat) = 1 - eps*rstd^2` — while leaving mean/rstd differentiable. `scale` is likewise
  value-pinned at `logit(saved gate)` through a detached shift, so every gradient path stays live.
  Gradients are all taken from autograd, never hand-derived: hand-deriving
  `dscale = dy * x_norm * g * (1-g)` is a chance to make the same slip the kernel might have made
  and then agree with it.

Coverage of the runnable set is now complete: **99 of 103 kernels have a numeric reference, 0 lack
a driver, and the 4 without a checker cannot launch on this card.**

## Not covered

* Constexpr flag combinations the drivers do not reach (`APPLY_MASK=True`, `SAVE_XN=True`,
  `FUSE_STATS=True`, `ADD_RESIDUAL=True` outside trimul) are separate compilations and are not in
  any sweep above.
* sm90 / sm100 kernels (6 cute collectives) cannot run on sm86 and are untested here, as is
  `transition_cuda_kernel.cu`, which no Python code imports.
### The three kernels that had no driver at all

`transition_cast_cuda`, `transition_swiglu_cuda`, `transition_bwd_cuda` — the `cast_kernel`,
`swish_mul_kernel` and `transition_grad_kernel` of
`transition/cuda/transition_cuda_kernel.cu` — were reported `untested` with no driver because
nothing in the package imports that extension: it is built only by the standalone
`transition/cuda/setup.py` as `transition_cuda_ext_v2`, and the `transition_cuda_b2b` setting
refers to the *other* extensions loaded in `transition/cuda/__init__.py`.

"No import path" is a reason a kernel cannot be *reached*, not a reason it cannot be *tested*. The
sources are in the tree, so the driver JIT-loads them the way the sibling `__init__.py` loads its
own (inside the driver, not at module scope, so a build failure is charged to these three kernels
instead of breaking every driver in the module), and drives them through the two functions the
`.cpp` exports. All three kernels are reachable from those: `swish_mul_kernel` via
`launch_swish_mul` (`.cu:325`), `transition_grad_kernel` via `launch_transition_grad` (`.cu:411`),
`cast_kernel` from the bf16<->fp32 conversion around the cuBLAS GEMM (`.cu:161`, `:170`). Shapes
are quoted from the `TORCH_CHECK`s at `transition_cuda.cpp:45-68`, not invented.

They have checkers too, against a torch fp32 reference (autograd for the backward), because a
driver alone leaves them in exactly the state that let three masking bugs sit in this repo.

    aligned   cast/swiglu y=5.18e-03   bwd dwa=4.51e-03 dwb=4.39e-03 dws=4.09e-03 dx=4.88e-03
    ragged    cast/swiglu y=4.74e-03   bwd dwa=1.32e-02 dwb=1.48e-02 dws=4.29e-03 dx=4.47e-03

The backward's return order is taken from `transition_cuda_kernel.cu:480`
(`{dx, grad_a, grad_b, grad_squeeze}`) rather than inferred: shape cannot disambiguate it, because
`expand_a` and `expand_b` are both `(nN, N)`. The first version of the checker tried to map by
shape and correctly refused rather than guessing. Shape is still asserted as a cross-check, so a
future reordering surfaces as a mismatch instead of a silent swap.

**One characterised anomaly, left open.** `dwa`/`dwb` are 3x higher at the ragged extents while
`dx`/`dws` do not move. Three probes to place it:

* not M: at a fixed K=128, `dwa` is 3.9-4.8e-03 across M = 2048, 4090, 4093, 4096, 4099, 8189,
  8192, with no dependence on `M % 128`.
* K, but not a divisibility rule: K = 125 and 127 give 1.3-1.5e-02 while K = 124, 120, 128, 256 and
  **253** give ~4e-03. 253 has the same residue mod 8 as 125, so no `K % 8` rule explains it.
* the numerator, not the denominator: `rel` is `max|a-e| / max|e|`, and splitting it shows
  `max|e|` unchanged (~26000) while `|err|` triples (130 -> 380). Per unit of K the clean cases sit
  at 0.97-1.02 and K = 125/127 at 2.95-3.04.

So it is a real numerical degradation localised to K just below 128, not a normalisation artifact.
It stays 3.5x inside the acceptance band, so it is a precision anomaly rather than a wrong answer,
and this extension has no caller in the repo. Root cause not identified; the `expand_a_work` /
`squeeze_work` staging copies in the forward are the place to look.

* The rename orphaned 128 tuned cache JSONs; 109 map to a current op through
  `docs/kernels/rename-map.tsv`, but all 109 carry a `config_space_hash` that no longer matches the
  current single-row config space, so `select_config` would reject them as stale. They remain in
  git history. sm80 / sm90 / sm100 tuning cannot be regenerated on this cluster.
