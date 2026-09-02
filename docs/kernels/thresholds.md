# Kernel thresholds: every numeric literal that decides something

Derived, not hand-written. An AST pass over `src/miniworld_engine/kernels/**` collects numeric
literals appearing in comparisons **outside** `@triton.jit` bodies — inside a kernel a constexpr
comparison is tile algebra, not policy. 0/1/2/-1 and anything under 8 are skipped, as are 100
and 1000. Script: [`audit_thresholds.py`](audit_thresholds.py), run from the repo root.

**48 sites.** The audit was looking for numbers that route work without saying why. Most of these
turn out not to be that. Three were. Two of those turned out to be documented
elsewhere in their own file, and the third — the only one reachable on our hardware — is now
sourced to the module that measured it, after the branch it guards was found to have never run.
What is left is two thresholds that are genuinely unjustified and genuinely unmeasurable here,
because both are SM90-only and this cluster has no Hopper.

## What each kind is, and how many

| kind | n | is it a problem? |
|---|---|---|
| architecture predicate | 23 | No. Needs a name, not a measurement. |
| performance threshold, evidence recorded | 12 | No. Three of these were miscounted at first — see below. |
| support / correctness bound | 3 | No. Past it the call raises or is wrong, not slow. |
| kernel hint | 2 | No. Changes a codegen hint, not a code path. |
| hardware limit | 2 | No. A number the chip fixes. |
| dtype width test | 2 | No. `width == 16` is "is this 16-bit". |
| tile sizing | 1 | No. Picks a tile that covers the shapes, and says so. |
| config discriminator | 1 | No. Picks a paired constant, and explains itself at length. |
| **performance threshold, no evidence** | **2** | **Both SM90-only, and this cluster has no Hopper.** |
| total | 48 | |

Three were given names this pass (2 support bounds and 1 threshold); the counts above are after
that, so those three no longer appear as bare literals.

## The 5 that route work — re-read, one at a time

The first pass read +/-8 lines around each site and called all five unjustified. Reading the
whole enclosing file changes the verdict on two of them, and shows a second thing worth knowing:
whether the branch is reachable at all on the hardware and dtype we actually run.

| file | line | value | evidence | reachable here? |
|---|---|---|---|---|
| `conditioned_transition/triton/training.py` | 447 | 8192 | **Yes**, 30 lines up | **No** — bf16 forces cuBLAS |
| `conditioned_transition/triton/training.py` | 448 | 512 | **Yes**, 30 lines up | **No** — bf16 forces cuBLAS |
| `transition/whole_op.py` | 41 | 256 | No | No — SM90 only, and this cluster has no H100 |
| `transition/whole_op.py` | 49 | 256 | **Now yes** — sourced to the module | **Yes**, once the branch was fixed — it used to raise |
| `transition/triton/fused.py` | 1467 | 128 | No | No — SM90 only, and off by default (`transition_dab_lnbwd`) |

### The two in `training.py` are documented, just not where the audit looked

`_pick_fwd`'s docstring says only "Measured-best forward backend per regime". The figures are on
the `_FWD_MODE` constant above it:

> MEASURED (H100, CUDA-graph fwd+bwd; identical backward, only the forward differs): both beat
> eager 1.12-1.28x. fused-vs-cublas: fused WINS atom large-M (8192: 1.03x) and small token
> (384/512: 1.01-1.02x), ~ties mid, REGRESSES token>=768 (0.93-0.95x) because the forward must
> additionally write ab,h,out,scale (saved-for-bwd) that inference never writes.

Card, harness, direction and margin, plus a mechanism for the regression. That is a real
justification. Two caveats stand: the margins are 1.01-1.03x, which is close to noise for a
routing decision; and it is an H100 result with no per-card guard — the same shape as the `adaln`
bug. What keeps it harmless is 30 lines below, where `x.dtype == torch.bfloat16 and mode ==
"fused"` is rewritten to `"cublas"` because the fused kernel's register pressure was never
benchmarked at bf16. bf16 is our default dtype, so **at the dtype we run, `_pick_fwd`'s two
thresholds decide nothing**. They govern fp32 only.

Fix is a docstring cross-reference, not a measurement.

### `whole_op.py:41` and `fused.py:1467` cannot be settled on this cluster

Both are behind an SM90 test and `sinfo` shows only A6000 and A5000 — no Hopper. `fused.py:1467`
is additionally behind a setting that is off by default and its own comment calls it
"Experimental/gated". Unjustified, and unmeasurable here; leaving them flagged is the honest
outcome, not a to-do that quietly never closes.

### `whole_op.py:49` — settled, and not the way the audit expected

Chasing it found the branch it guards **had never run**. It passed
`x.reshape(-1, d_hidden)` to `triton_layernorm`, which reads the shape it is given to build its
autotune key; `rows_of` rejects an already-flat shape by design, because once a token
`(B, L, D)` and a pair `(B, L, L, D)` are both 2-D it cannot tell which key to build. On an
A5000, `ops.transition` raised `ValueError` at all four of krystal's widths. Nothing caught it
because nothing called it: `modules/transition/module.py` — what the model runs — carries its own
copy of the arch dispatch and reaches `kernels.triton_transition` directly, and the tests naming
`ops.transition` check the export table and the compile surface, not a call.

So the threshold never needed its own measurement. It needed a **source**, and the module is it:

> A100, cudagraph-manual: fused 2.28 ms vs split 1.40 at d=256, and 10.9 vs 4.8 at d=512, while
> d=128 (the AF3 shape) is where fused wins. sm_86 (A5000/A6000) needs its own answer — there the
> fused path loses at EVERY d, 0.88-0.95x across both the L and d sweeps — so on sm_86 all widths
> take the split.

The wrapper had only the first clause, so even working it would have sent d=128 to the fused path
on an A5000, against that measurement. Both clauses now match the module, quoted at the site, and
`tests/numerics/test_whole_op_reachable_gpu.py` calls `ops.transition` at 16, 128, 384 and 768 so
the branch cannot go dead again unnoticed. 4 passed on an A5000; the 384 and 768 cases raise on
the old code.

An independent A/B on this branch was attempted and **abandoned, on purpose**. The tuned cache
does not match this branch's config grid ("STALE (kernel config grid changed)"), so both paths
fall back to a heuristic 24-of-1296 subset. A first row at d=16 came back with fused ahead 1.27x,
which contradicts the sm_86 clause — but an untuned eager measurement cannot be set against the
module's tuned cudagraph-manual sweep, and reporting it as if it could is the exact error this
audit exists to remove. Settling sm_86 independently needs a tuned cache for this branch first.

## Named this pass

| where | was | now | why |
|---|---|---|---|
| `adaln/triton/inference.py` | `x.shape[-1] <= 256` | `_FUSED_D_MAX` | The only one of the 5 measured so far. A5000 bf16 numbers are in the comment, and so is what is still unknown: the 256/384/512 boundary, and every card that is not an A5000. Its previous justification (1.12–1.21x) measured the SM90 cute path, which an Ampere card never takes. |
| `augmented_attention/triton/main.py` | `D > 64` | `_MAX_HEAD_DIM` | Support bound, not routing — past it the call raises. Set by the tile the kernel is written around, not by a measurement. |
| `augmented_attention/triton/memory_efficient.py` | `D > 64` | `_MAX_HEAD_DIM` | Same. |

Two `# noqa: PLR2004` suppressions went away with these. The three that remain sit on `== 9`
and `== 10`, which are architecture tests.

## Architecture predicates (23)

`capability()[0] == 9` and friends. The number is a hardware generation, so it cannot drift and
needs no measurement. What it needs is a **name**: `modules/dispatch.py` already defines
`is_sm90`, `is_sm90plus`, `is_sm100` and `is_sm86`, and these sites spell the same test out by
hand, so a new architecture has to be found in all of them.

Left alone deliberately. A blind sweep of 23 sites across five kernel families is more risk than
it buys with no GPU to verify on, and `_compile.py` warns that calling `get_device_capability`
inside an `@opaque` returns a non-Tensor — some of these may be inline for that reason.

| file | line | value | expression |
|---|---|---|---|
| `adaln/triton/inference.py` | 538 | 9 | `torch.cuda.get_device_capability(x.device)[0] == 9` |
| `bias_only_attention/dispatch.py` | 92 | 9 | `torch.cuda.get_device_capability(idx)[0] == 9` |
| `layernorm_linear/__init__.py` | 62 | 9 | `torch.cuda.get_device_capability(x.device)[0] == 9` |
| `layernorm_linear/cute/dgrad_lnbwd.py` | 218 | 9 | `dev[0] == 9` |
| `layernorm_linear/cute/dgrad_lnbwd.py` | 264 | 9 | `dev[0] == 9` |
| `layernorm_linear/cute/gemm_layernorm_linear.py` | 193 | 8 | `device_capacity[0] == 8` |
| `layernorm_linear/cute/gemm_layernorm_linear.py` | 257 | 9 | `device_capacity[0] <= 9` |
| `layernorm_linear/cute/gemm_layernorm_linear_fused.py` | 851 | 9 | `device_capacity[0] == 9` |
| `transition/cute/backward_gatebwd.py` | 150 | 90 | `self.arch == 90` |
| `transition/cute/backward_gatebwd.py` | 252 | 9 | `device_capacity[0] == 9` |
| `transition/cute/backward_gatebwd.py` | 309 | 9 | `device_capacity[0] == 9` |
| `transition/cute/dab_lnbwd.py` | 254 | 9 | `dev[0] == 9` |
| `transition/cute/dab_lnbwd.py` | 309 | 9 | `dev[0] == 9` |
| `transition/cute/gemm_transition_swiglu.py` | 97 | 90 | `self.arch == 90` |
| `transition/cute/gemm_transition_swiglu.py` | 187 | 9 | `device_capacity[0] == 9` |
| `transition/cute/gemm_transition_swiglu.py` | 247 | 9 | `device_capacity[0] == 9` |
| `transition/triton/fused.py` | 1189 | 10 | `_cap_major == 10` |
| `transition/triton/fused.py` | 1190 | 9 | `_cap_major == 9` |
| `transition/triton/fused.py` | 1214 | 10 | `torch.cuda.get_device_capability(x2.device)[0] == 10` |
| `transition/triton/fused.py` | 1371 | 10 | `torch.cuda.get_device_capability(x2.device)[0] == 10` |
| `transition/triton/fused.py` | 1432 | 9 | `torch.cuda.get_device_capability(x2.device)[0] == 9` |
| `transition/triton/fused.py` | 1468 | 9 | `torch.cuda.get_device_capability(x2.device)[0] >= 9` |
| `trimul_inproj/whole_op.py` | 95 | 10 | `torch.cuda.get_device_capability(x.device)[0] >= 10` |

## Performance thresholds that DO carry evidence (12)

Recorded here so the audit does not have to be redone to find out they are fine.

| file | line | value | the evidence in the code |
|---|---|---|---|
| `layernorm/compile_native.py` | 43 | 128, 512 | "Hand-CUDA warp-per-row bwd beats triton 1.2–1.46x for bf16 128<=N<=512 (measured H100)". H100-measured, but **only reached on H100** (`cc == _HOPPER`) or when calibration is off; otherwise the dispatcher times the real paths at runtime and caches the winner. The H100 number never governs another card. |
| `layernorm/compile_native.py` | 45 | 384 | Same function, same runtime-calibration guard. |
| `layernorm/compile_native.py` | 90 | 128, 512 | Decides which impls enter the *calibration* set, so a wrong bound costs a candidate, not a wrong choice. |
| `layernorm/triton/main.py` | 396 | 128, 512 | "CUDA beats triton 1.17x @L512, 1.28x @L1024" for the masked path; dense is neutral and stays behind an env flag. Both sides of the split stated. |
| `layernorm_linear/autograd.py` | 156 | 128 | "wins/ties at K=128 across M (A/B: 1.29x@16384, ~1.0x at larger M)", plus *why* K=256 loses: the full-N epi subtile starves the mainloop with D+C both in smem. |
| `transition/whole_op.py` | 49 | 256 | The A100 and sm_86 sweeps in `modules/transition/module.py`, now quoted at the site. |
| `conditioned_transition/triton/training.py` | 447, 448 | 8192, 512 | The H100 sweep quoted above, on the `_FWD_MODE` constant 30 lines up rather than on `_pick_fwd` itself. Unreachable at bf16. |
| `transition/cute/gemm_transition_swiglu.py` | 197 | 128 | "the win the K-sweep found: K<=128 -> 256x128 coop; K>=256 -> 192x128 pingpong", verified 2026-08-04 at cos=1.0. A cache-miss fallback, not the primary path. |

## Everything else (11)

Counts as: 3 support/correctness bounds, 2 kernel hints, 2 hardware limits, 2 dtype width tests,
1 tile sizing, 1 config discriminator.

Not thresholds. Listed so a future audit does not re-flag them.

| file | line | value | kind |
|---|---|---|---|
| `layernorm_linear/triton/fused.py` | 188 | 1024 | correctness bound — "one-block-K assumption broken; correctness-first eager fallback" |
| `triangle_attention/triton/atomic.py` | 630 | 32 | support bound — raises `ValueError` |
| `transition/triton/fused.py` | 1092 | 512 | support bound — the hand-CUDA lnbwd requires bf16/fp16 + K<=512 + contiguous; its perf claim (1.10x, 326 vs 358us at K=128) is stated separately |
| `trimul_inproj/cute/back_split.py` | 36 | 128 | tile sizing — `tile_n=128` covers N∈{128,256,512}, N=64 gets 64; explained in the docstring |
| `layernorm_linear/triton/mmajor_bwd.py` | 204 | 128 | kernel hint — feeds `VEC_HINT`, not a path choice |
| `layernorm_linear/triton/mmajor_bwd.py` | 257 | 128 | kernel hint — same |
| `tm1/cute/_blackwell_dense_gemm.py` | 1232 | 16 | hardware limit — max CTAs per cluster |
| `tm1/cute/sm100_gate_gemm_collective.py` | 102 | 512 | hardware limit — the Blackwell TMEM column budget |
| `transition/cute/backward_gatebwd.py` | 148 | 16 | dtype width — "is this a 16-bit type" |
| `transition/cute/gemm_transition_swiglu.py` | 95 | 16 | dtype width — same |
| `drivers/conditioned_transition.py` | 89 | 128 | config discriminator — `_DC_BASE = 384 if _D_BASE > 128 else 128`, picking the d_cond the model actually pairs with each d_hidden; carries a seven-line explanation |
