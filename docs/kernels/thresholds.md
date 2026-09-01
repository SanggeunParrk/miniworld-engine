# Kernel thresholds: every numeric literal that decides something

Derived, not hand-written. An AST pass over `src/miniworld_engine/kernels/**` collects numeric
literals appearing in comparisons **outside** `@triton.jit` bodies — inside a kernel a constexpr
comparison is tile algebra, not policy. 0/1/2/-1 and anything under 8 are skipped, as are 100
and 1000. Script: [`audit_thresholds.py`](audit_thresholds.py), run from the repo root..

**48 sites.** The point of the audit was to find numbers that route work without saying why. Most
of these turn out not to be that, and the count of the ones that are is **5**.

## What each kind is, and how many

| kind | n | is it a problem? |
|---|---|---|
| architecture predicate | 23 | No. Needs a name, not a measurement. |
| performance threshold, evidence recorded | 7 | No. |
| support / correctness bound | 4 | No. Past it the call raises or is wrong, not slow. |
| kernel hint | 2 | No. Changes a codegen hint, not a code path. |
| hardware limit | 2 | No. A number the chip fixes. |
| dtype width test | 2 | No. `width == 16` is "is this 16-bit". |
| config discriminator | 1 | No. Picks a paired constant, and explains itself at length. |
| **performance threshold, no evidence** | **5** | **Yes. This is the backlog.** |
| named this pass | 3 | Was 2 support bounds + 1 unjustified threshold. |

## The 5 that route work and do not say why

| file | line | value | expression | what it claims |
|---|---|---|---|---|
| `conditioned_transition/triton/training.py` | 447 | 8192 | `M >= 8192` | fused beats cuBLAS above this atom M |
| `conditioned_transition/triton/training.py` | 448 | 512 | `M <= 512` | fused beats cuBLAS below this token M |
| `transition/whole_op.py` | 41 | 256 | `d_hidden >= 256` | wide-d goes to the cute fused expand (SM90) |
| `transition/whole_op.py` | 49 | 256 | `d_hidden >= 256` | wide-d goes to the split GEMM (pre-Hopper) |
| `transition/triton/fused.py` | 1467 | 128 | `K <= 128` | the cute dab+lnbwd fusion wins below this K |

The two in `training.py` sit under a docstring that says "Measured-best forward backend per
regime", which is a claim of measurement with no number, no card and no date attached — the same
shape as the `adaln` 256 turned out to be, where the recorded justification was an H100 result
governing Ampere routing. Treat "measured" without a figure as unverified.

Each needs one A/B on both sides of the boundary, per card. That is one GPU job for the pair in
`training.py` (same function), one for `whole_op.py`, one for `fused.py`.

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

## Performance thresholds that DO carry evidence (7)

Recorded here so the audit does not have to be redone to find out they are fine.

| file | line | value | the evidence in the code |
|---|---|---|---|
| `layernorm/compile_native.py` | 43 | 128, 512 | "Hand-CUDA warp-per-row bwd beats triton 1.2–1.46x for bf16 128<=N<=512 (measured H100)". H100-measured, but **only reached on H100** (`cc == _HOPPER`) or when calibration is off; otherwise the dispatcher times the real paths at runtime and caches the winner. The H100 number never governs another card. |
| `layernorm/compile_native.py` | 45 | 384 | Same function, same runtime-calibration guard. |
| `layernorm/compile_native.py` | 90 | 128, 512 | Decides which impls enter the *calibration* set, so a wrong bound costs a candidate, not a wrong choice. |
| `layernorm/triton/main.py` | 396 | 128, 512 | "CUDA beats triton 1.17x @L512, 1.28x @L1024" for the masked path; dense is neutral and stays behind an env flag. Both sides of the split stated. |
| `layernorm_linear/autograd.py` | 156 | 128 | "wins/ties at K=128 across M (A/B: 1.29x@16384, ~1.0x at larger M)", plus *why* K=256 loses: the full-N epi subtile starves the mainloop with D+C both in smem. |
| `transition/cute/gemm_transition_swiglu.py` | 197 | 128 | "the win the K-sweep found: K<=128 -> 256x128 coop; K>=256 -> 192x128 pingpong", verified 2026-08-04 at cos=1.0. A cache-miss fallback, not the primary path. |

## Everything else (11)

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
