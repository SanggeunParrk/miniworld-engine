# Fused Transition forward and backward on H100

The pair Transition backward (`y = x + W_s(silu(W_a·LN(x)) · (W_b·LN(x)))`, D = 128, H = 4D = 512, bf16) runs today as five
launches: a cuBLAS `dh`, a Triton gate backward, two cuBLAS weight-gradient GEMMs, a cuBLAS `d_xn` and a LayerNorm backward.
Two thirds of that time is intermediate tensors going to HBM and coming back — the backward moves about 1.66 GB where the
essential inputs and outputs are 120 MB. This experiment replaces the whole thing with **one CUDA kernel** (plus a small
partial-sum reduction) that keeps `dh`, `h`, `dA` and `dB` in registers and shared memory and never writes them out.

The forward has the same problem and the same fix: it runs as three kernels (LayerNorm, expand-SwiGLU, squeeze-residual) and
puts the `[M][512]` SwiGLU activation through HBM twice — 151 MB each way at L384. `src/transition_fwd.cu` fuses it into one
kernel that keeps that activation in registers and still emits what the backward needs (`xn`, `rstd`, `c1`).

Both are now **wired into the engine**: `modules.Transition` routes to them wherever the shape fits. See "Wiring it up"
for what the dispatch looks like and what it deliberately does not take.

![The two CTA roles and where each gradient is produced](wiring.svg)

## Result

What a training run gets is in [What the wired path measures](#what-the-wired-path-measures): 1.9x the Triton residual path,
2.2x `torch.compile`, 3.2x eager PyTorch, through the real dispatch. The rest of this section is the two ops on their own,
which is where the design decisions were made and where the numbers are larger.

node02 H100 80 GB, CUDA 12.9, PyTorch 2.10 cu128, same session, CUDA-graph replay median (`records/bench-L{384,768}.json`):

| | L384 (M = 147 456) | L768 (M = 589 824) |
|---|---:|---:|
| engine backward (5 launches) | 821 µs | 3113 µs |
| **this kernel** (1 launch + reduction) | **406 µs** | **1530 µs** |
| speed-up | **2.02×** | **2.03×** |
| tensor floor of this design (22·M·D·H) | 224 µs | 897 µs |
| absolute tensor floor (16·M·D·H) | 163 µs | 652 µs |

132 CTAs × 256 threads, 231 KB shared memory, 255 registers, no spill, no cooperative launch and no cluster.

The forward, same session (`bench_fwd.py`, `records/fwd-L{384,768}.json`):

| | L384 | L768 |
|---|---:|---:|
| engine forward (3 launches) | 272 µs | 1031 µs |
| **this kernel** | **130 µs** | **482 µs** |
| speed-up | **2.10×** | **2.14×** |
| tensor floor (6·M·D·H) | 61 µs | 244 µs |

Inference and training take the *same* forward on this branch: with the default
`transition_residual_fusion` both `_inference_forward` and `_training_forward` route to `transition_residual`, and the two
measure 278 and 280 µs at L384. (Turning residual fusion off does split them — 199 µs inference, 231 µs training through
the hand-CUDA b2b — and is faster on the forward alone, which is worth a look on its own.) This kernel serves both; built
with `-DFWD_SAVE=0` it drops the `xn` / `rstd` / `c1` stores that only the backward needs:

| forward | L384 | L768 |
|---|---:|---:|
| training (saves `xn`, `rstd`, `c1`) | **130 µs** | **482 µs** |
| inference (`-DFWD_SAVE=0`) | 139 µs | 521 µs |

`-DFWD_SAVE=0` used to be the faster of the two and no longer is: after the LayerNorm restructure the build that writes
`xn` schedules better than the one that does not. One build serves both cases.

Against Anthropic's own kernels at this width (`records/vs-anthropic.md`, one process, same timing, forward only): their
best row is Triton `v2` at 157.9 µs (L384) / 596.9 µs (L768) — they ship **no CUDA Transition kernel at c = 128**, and no
Transition backward at all — and the engine's fastest path on the parity checkout is 156.8 / 563.0 µs. This kernel is
129.4 / 494.4 µs, 1.22× and 1.21× their best, and the most accurate of the six. The larger multiples quoted above are against `main`, whose default
routes both inference and training to the three-kernel path; part of that win is recovering the difference between branches.

And the two together at the module level — `miniworld_engine.modules.Transition(128, 4)` in bf16 training, forward + backward
(`bench_module.py`, `records/module-L{384,768}.json`):

| | L384 | L768 |
|---|---:|---:|
| engine module fwd + bwd | 1109 µs | 4106 µs |
| with this backward only | 738 µs (1.50×) | 2747 µs (1.49×) |
| **with both** | **583 µs (1.90×)** | **2307 µs (1.78×)** |

The two ops sum to 536 µs at L384 and the module measures 583: the remaining ~47 µs is the autograd boundary, the reshapes
and the launch gaps, and it does not shrink. That is why two ~2.1× ops make a 1.9× module.

Gradients through the real module agree with the engine path to 4.2e-5 (`dgamma`) … 5.9e-4 (`dW*`), against an engine
run-to-run noise of 0 … 1.3e-6. Note that the module zero-initialises the squeeze weight, which makes the backward
degenerate (`dh = 0`, so four of the five parameter gradients are exactly zero in *both* paths); `bench_module.py` gives it a
trained-looking value first, or the comparison would be comparing zeros.

Numerics: all six gradients carry the same relative RMS error against an fp32 autograd reference as the engine's own backward
does — `dx` 3.07e-3, `dWa` 3.88e-3, `dWs` 3.42e-3 at L384, and the engine's row is printed next to it by `bench.py --engine`.
Against the engine directly the difference is accumulation order only: `dx` 5.4e-5 (0.02 % of elements differ), `dW*` 2.0-2.5e-4.
The bf16 rounding points and the multiplication order of the contract are unchanged. **All six outputs are bit-reproducible**
run to run: there is no atomic anywhere in the kernel.

## How it works

Two CTA roles in one launch, following the split the B1-B4 TriMul training kernel uses.

**Weight CTAs** (8 hidden slices × `DW_REPL` replicas, 64 CTAs at the default) own a 64-unit hidden slice and keep its
`W_s`, `W_a`, `W_b` resident in shared memory, so no weight streaming. Both warpgroups run the gate stage on their own 64 rows
of a 128-row tile and write `h`, `dA`, `dB` to shared memory; then they split the weight gradients over the whole tile — WG0
takes `dW_a` and the left half of `dW_s^T`, WG1 takes `dW_b` and the right half — as 96 fp32 accumulators per thread.

**Input CTAs** (the remaining 68) stream all eight 64-unit hidden chunks of their 128-row tile through a two-slot TMA weight
ring. Per chunk each warpgroup computes `dh` and the packed `[a|b]`, forms `dA` and `dB` as bf16 **directly in the wgmma
A-fragment registers** — the m64n64 C fragment is exactly the m64k64 A fragment, so there is no shared-memory round trip and
no `stmatrix` — and accumulates `d_xn` with an RS m64n128 over the same packed `[W_a; W_b]` operand. `d_xn` never leaves
registers: the LayerNorm backward, the residual, `dx` and the `dgamma` / `dbeta` partials all run in that fragment layout.

Only `dh`, `a` and `b` are recomputed, by the weight role, so the kernel executes 22·M·D·H FLOP against a 16·M·D·H minimum.
That 1.375× is the price of having no cross-CTA communication at all, and it is a bargain: the first design avoided the
recomputation by splitting the hidden axis over a cluster and reduce-scattering the `d_xn` partial sums through distributed
shared memory, and was 2.5× slower because the two cluster barriers and the remote stores a tile needs cost 572 µs of its
1017 (`records/progression.md`).

The forward kernel is the same skeleton without the two roles: a persistent CTA per 128-row tile, the same two-slot TMA ring
over the eight hidden chunks, LayerNorm computed in registers and written to shared memory (and straight to global for the
backward, one fully coalesced 256-byte row per warp), `[a|b]` as one m64n128 chain over the packed `[W_a; W_b]` tile, the
SwiGLU result formed as bf16 in the m64k64 A-fragment registers, and `acc += h_j · W_s^T_j` as an RS m64n128 accumulating in
registers across all eight chunks. `W_s` is passed transposed so the squeeze operand is `[K = hs][N = d]`.

`DW_REPL` is the only tuning knob and it is compiled in. At 8 the two roles are within 5 % of each other and 1152 tiles divide
exactly; the sweep is in `records/ratio-r*-L384.json`.

## Layout

| | |
|---|---|
| `src/transition_bwd.cu` | the backward kernel and the partial-sum reduction; the only include is the Anthropic v5 device header set |
| `src/transition_fwd.cu` | the forward kernel: LayerNorm + expand + SwiGLU + squeeze + residual, emitting `xn` / `rstd` / `c1` |
| `build.sh`, `build_fwd.sh` | `[OUT=<name>] ./build.sh [-DDW_REPL=<R>]` → `build/<OUT>.cubin` (nvcc, sm_90a, compute node only) |
| `bench_vs_anthropic.py` | the same forward against Anthropic's own Transition rows and the engine's paths, one process, one timing method |
| `bench.py`, `bench_fwd.py` | correctness against an fp32 reference, bit-reproducibility, CUDA-graph timing, `--engine` for the baseline, `--save` for a record |
| `drv.py` | minimal `cuda.bindings` launcher: TMA descriptors, cubin load, by-value argument packing |
| `bench_module.py` | the same, at the module level: times `modules.Transition` fwd + bwd with and without this backward, and checks the gradients agree |
| `bench_wired.py` | the shipped dispatch: `modules.Transition` with `transition_fused_sm90a` on and off, plus PyTorch eager and `torch.compile` over the same weights |
| `verify_package.py` | CPU-only integrity checks (vendored-header hashes, portable paths, recorded measurements) |
| `records/` | the measurements above, the ratio sweep, and `progression.md` — how it got here and the eleven things that did not work |

The Anthropic v5 device primitives (`tmn_ptx.cuh`, `tmn_kernels.cuh`, `common/tmn_math.cuh`, Apache-2.0, upstream revision
`f4f62fa`) are not duplicated *here*: this experiment's build includes the copy vendored for the sibling experiment at
`../trimul_b7b12/vendor/anthropic_v5/csrc`, and `verify_package.py` pins their hashes. The wired copy in
`src/miniworld_engine/kernels/transition/cuda/anthropic_v5/` is the same three files, inside the package so the shipped
kernel builds from an installed engine rather than only from a checkout.

```
OUT=transition_bwd_r8 ./build.sh -DDW_REPL=8   &&  python bench.py       --length 384 --dw-repl 8 --engine
./build_fwd.sh                                 &&  python bench_fwd.py   --length 384 --engine
python bench_module.py --length 384                       # both, through the real module
python bench_wired.py  --length 384                       # the shipped dispatch, vs Triton and PyTorch
```

Needs a compute node (nvcc and the GPU), torch with CUDA and `cuda.bindings`.

## Wiring it up

Both kernels live in the engine as of this branch, at
`src/miniworld_engine/kernels/transition/cuda/`:

| file | what it is |
|---|---|
| `transition_fused_fwd_sm90a_kernel.cu` | this experiment's `src/transition_fwd.cu`, plus a host launcher |
| `transition_fused_bwd_sm90a_kernel.cu` | this experiment's `src/transition_bwd.cu`, plus host launchers |
| `transition_fused_sm90a.cu` | torch bindings: TMA descriptors, output allocation, stream |
| `fused_sm90a.py` | the JIT build, the shape gate, the two opaque ops and the autograd Function |
| `anthropic_v5/` | the three vendored Anthropic device headers the kernels are written against |

`modules.Transition._residual_forward` is the single dispatch point. It asks `fused_sm90a.available`, and takes the fused
path only when every one of the kernel's own requirements holds: sm_90, bf16, `d_hidden == 128`, hidden 512 (`n == 4`), and a
row count that is a whole number of 128-row tiles. Everything else keeps the Triton residual path, unchanged. Turning it off
is `transition_fused_sm90a=False` or `MINIWORLD_TRANSITION_FUSED_SM90A=0`.

Four things the wiring had to settle, beyond copying the sources across.

1. **The persistent grid is compiled, not chosen.** `NCTA` is one CTA per SM and it is a build-time constant, so the
   extension is built per SM count (`_ext(ctas, dw_repl, save)`) and the module name carries it. A second card with a
   different multiprocessor count gets its own build instead of a grid that silently does not cover the device.
2. **Inference gets its own build.** Writing `xn`, `rstd` and `c1` sits inside the LayerNorm epilogue, so it cannot be a
   runtime flag. `FWD_SAVE` selects the variant and the autograd Function picks it from `ctx.needs_input_grad`; the binding
   asserts that the flag it was passed matches the build, which is what stops an inference build writing through the
   1-element placeholders.
3. **Both launches are opaque ops.** The launch is split out of the autograd Function and registered through `opaque`, the
   same way every other engine kernel is, so Dynamo traces through the Function and stops only at the launch. Without that
   the residual path would graph-break and inductor's cudagraph trees would bail on the whole region.
4. **TMA descriptors are cached, not rebuilt.** A descriptor is 128 opaque bytes tied to one base pointer, so it is keyed on
   (pointer, dims, box). Weights hit the cache after the first step; activations re-encode, which is a host-side memcpy and
   does not touch the GPU. `cuTensorMapEncodeTiled` is resolved through `cudaGetDriverEntryPoint` so the extension does not
   have to link `libcuda`.

What the wiring does **not** do, and should be looked at before this is relied on for a long run:

- **The engine's own parity tests are not re-baselined.** Nothing outside `test_transition_fused_sm90a_gpu.py` has been
  re-recorded, and switching this on changes recorded outputs by the amounts in the table above. That deserves a deliberate
  re-baseline rather than an accidental one.
- **Nobody has run a training-convergence check.** Two launches computing the same function to the same tolerance is not the
  same claim as a run that converges the same way.

### What the wired path measures

`bench_wired.py` times the dispatch as a training run gets it: the same `modules.Transition`, the same settings object, the
only difference being `transition_fused_sm90a`. Next to it, the same op in plain PyTorch over the same weights, through the
engine's own `transition_pytorch` reference. node01 H100 80 GB, forward plus backward, `records/wired-L{384,768}.json`:

| | L384 | ×  | L768 | × |
|---|---:|---:|---:|---:|
| PyTorch eager | 1822 µs | 3.24 | 6853 µs | 3.28 |
| PyTorch, `torch.compile` | 1211 µs | 2.15 | 4523 µs | 2.16 |
| Triton residual path (`transition_fused_sm90a=False`) | 1074 µs | 1.91 | 4022 µs | 1.92 |
| **fused sm_90a (default)** | **562 µs** | — | **2092 µs** | — |

That is the whole module, so it is bounded by how much of it these two kernels are; the op-level numbers above are larger.
Several Triton ops on the baseline side fall back to heuristic configs because this worktree has no tuned autotune cache for
this card, which if anything flatters the baseline's *variance*, not its median -- the same run on node02 before the wiring
measured 1109 µs for the same path.

### Accuracy

Against the Triton path directly the output differs by 2.5e-3 relative, which looks alarming and is not: both paths round to
bf16 in the same places but not in the same order, so they sit about that far apart while each sits about that far from
fp32. `tests/numerics/test_transition_fused_sm90a_gpu.py` asserts the claim that means something -- the fused path is no
further from an fp32 run of the same module than the Triton path is, for the output, `dx` and all five parameter gradients.
Direct agreement, L384:

| | out | dx | dgamma | dbeta | dWa | dWb | dWs |
|---|---:|---:|---:|---:|---:|---:|---:|
| rel_rms vs Triton | 2.5e-3 | 9.9e-5 | 7.7e-5 | 8.2e-5 | 3.5e-4 | 3.6e-4 | 4.0e-4 |
| rel_rms vs PyTorch eager | 4.0e-3 | 4.5e-3 | 5.1e-3 | 6.0e-3 | 4.0e-3 | 4.2e-3 | 4.5e-3 |

The PyTorch row is larger for the same reason and more of it: eager bf16 materialises every intermediate at bf16, so it is
the furthest of the three from fp32, not a yardstick to be close to.

Eight tests cover the gate (what it rejects and why), the env switch, both accuracy claims, that the module really
dispatches to it, and that a replay is bit-identical.

## Headroom

Not a lever: recomputing `xn` in the backward instead of saving it. It saves the forward's 9 µs (L384) / 31 µs (L768) store
and saves the backward nothing — removing the `xn` read outright measures inside the noise, because the backward runs at
557 GB/s against a ~3 TB/s peak — while adding a LayerNorm-apply pass to the weight role, which is the binding one.
`records/progression.md` has the numbers. It remains the right trade if activation memory, not time, is the constraint.

The forward is at 47-51 % of its tensor floor after one round of tuning (`records/progression.md`): the LayerNorm's
reductions and the output stores are fixed, the weight stream and the transcendental measured free, and what is left is the
ring handshake (12 µs by ablation) and the wgmma drain between the two chains in a chunk. Software-pipelining that drain
needs a second `[a|b]` accumulator (64 registers on top of 183) and a three-deep ring for the squeeze operand, which the
freed shared memory would now allow.

The backward's tensor pipe is 58.1 % active (NCU `--set full`, L384). Since this design must execute 22·M·D·H FLOP, that *is* 406 µs;
330 µs would need 70 %. The stall profile is flat — the largest single SASS line is 4.5 % — so there is no hotspot left, and
five structural attempts in a row were neutral or worse. The two remaining levers are both blocked by a hard limit (registers
at 255, shared memory at the effective 231424-byte ceiling); `records/progression.md` has the details and the traps.
