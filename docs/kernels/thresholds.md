# Kernel thresholds: every numeric literal that decides something

Derived, not hand-written: an AST pass over `src/miniworld_engine/kernels/**` collecting numeric
literals in comparisons OUTSIDE `@triton.jit` bodies (a constexpr comparison inside a kernel is
the tile algebra, not a policy). 0/1/2/-1/0.5/100/1000 and anything under 8 are skipped.

Two kinds, and only one of them is a problem.

## Architecture predicates (27)

`capability()[0] == 9` and friends. The number is a hardware generation, so it cannot drift and
needs no measurement. What it needs is a NAME: `modules/dispatch.py` already defines `is_sm90`,
`is_sm90plus`, `is_sm100` and `is_sm86`, and these sites predate or ignore them, so the same
predicate is spelled out in many places and a new architecture has to be found in all of them.

| file | line | value | expression |
|---|---|---|---|
| `adaln/triton/inference.py` | 538 | 9 | `torch.cuda.get_device_capability(x.device)[0] == 9` |
| `bias_only_attention/dispatch.py` | 92 | 9 | `torch.cuda.get_device_capability(idx)[0] == 9` |
| `layernorm_linear/__init__.py` | 62 | 9 | `torch.cuda.get_device_capability(x.device)[0] == 9` |
| `layernorm_linear/cute/dgrad_lnbwd.py` | 218 | 9 | `dev[0] == 9` |
| `layernorm_linear/cute/dgrad_lnbwd.py` | 188 | 9 | `device_capacity[0] == 9` |
| `layernorm_linear/cute/dgrad_lnbwd.py` | 264 | 9 | `dev[0] == 9` |
| `layernorm_linear/cute/gemm_layernorm_linear.py` | 146 | 9 | `device_capacity[0] == 9` |
| `layernorm_linear/cute/gemm_layernorm_linear.py` | 249 | 9 | `device_capacity[0] <= 9` |
| `layernorm_linear/cute/gemm_layernorm_linear_fused.py` | 851 | 9 | `device_capacity[0] == 9` |
| `transition/cute/backward_gatebwd.py` | 252 | 9 | `device_capacity[0] == 9` |
| `transition/cute/backward_gatebwd.py` | 150 | 90 | `self.arch == 90` |
| `transition/cute/backward_gatebwd.py` | 228 | 9 | `device_capacity[0] == 9` |
| `transition/cute/backward_gatebwd.py` | 309 | 9 | `device_capacity[0] == 9` |
| `transition/cute/dab_lnbwd.py` | 254 | 9 | `dev[0] == 9` |
| `transition/cute/dab_lnbwd.py` | 204 | 9 | `device_capacity[0] == 9` |
| `transition/cute/dab_lnbwd.py` | 309 | 9 | `dev[0] == 9` |
| `transition/cute/gemm_transition_swiglu.py` | 187 | 9 | `device_capacity[0] == 9` |
| `transition/cute/gemm_transition_swiglu.py` | 97 | 90 | `self.arch == 90` |
| `transition/cute/gemm_transition_swiglu.py` | 164 | 9 | `device_capacity[0] == 9` |
| `transition/cute/gemm_transition_swiglu.py` | 247 | 9 | `device_capacity[0] == 9` |
| `transition/triton/fused.py` | 1189 | 10 | `_cap_major == 10` |
| `transition/triton/fused.py` | 1190 | 9 | `_cap_major == 9` |
| `transition/triton/fused.py` | 1214 | 10 | `torch.cuda.get_device_capability(x2.device)[0] == 10` |
| `transition/triton/fused.py` | 1371 | 10 | `torch.cuda.get_device_capability(x2.device)[0] == 10` |
| `transition/triton/fused.py` | 1468 | 9 | `torch.cuda.get_device_capability(x2.device)[0] >= 9` |
| `transition/triton/fused.py` | 1432 | 9 | `torch.cuda.get_device_capability(x2.device)[0] == 9` |
| `trimul_inproj/whole_op.py` | 95 | 10 | `torch.cuda.get_device_capability(x.device)[0] >= 10` |

## Performance thresholds (26)

These pick between two implementations, or bound what a kernel accepts. Each one is a claim that
something is faster (or unsupported) past that point, and almost none of them says where the
number came from. `adaln`'s was measured today and now records both what it found and what it
still does not know; the rest are as they were.

| file | line | value | expression |
|---|---|---|---|
| `conditioned_transition/triton/training.py` | 448 | 512 | `M <= 512` |
| `conditioned_transition/triton/training.py` | 447 | 8192 | `M >= 8192` |
| `drivers/conditioned_transition.py` | 89 | 128 | `_D_BASE > 128` |
| `layernorm/compile_native.py` | 45 | 384 | `n >= 384` |
| `layernorm/compile_native.py` | 43 | 128 | `128 <= n <= 512` |
| `layernorm/compile_native.py` | 43 | 512 | `128 <= n <= 512` |
| `layernorm/compile_native.py` | 90 | 128 | `128 <= n <= 512` |
| `layernorm/compile_native.py` | 90 | 512 | `128 <= n <= 512` |
| `layernorm/triton/main.py` | 396 | 128 | `128 <= N <= 512` |
| `layernorm/triton/main.py` | 396 | 512 | `128 <= N <= 512` |
| `layernorm_linear/autograd.py` | 156 | 128 | `K <= 128` |
| `layernorm_linear/cute/gemm_layernorm_linear.py` | 185 | 8 | `device_capacity[0] == 8` |
| `layernorm_linear/triton/fused.py` | 188 | 1024 | `K > 1024` |
| `layernorm_linear/triton/mmajor_bwd.py` | 257 | 128 | `K <= 128` |
| `layernorm_linear/triton/mmajor_bwd.py` | 204 | 128 | `K <= 128` |
| `tm1/cute/_blackwell_dense_gemm.py` | 1232 | 16 | `self.cluster_shape_mn[0] * self.cluster_shape_mn[1] > 16` |
| `tm1/cute/sm100_gate_gemm_collective.py` | 102 | 512 | `2 * single > 512` |
| `transition/cute/backward_gatebwd.py` | 148 | 16 | `args.mAuxOut.element_type.width == 16` |
| `transition/cute/gemm_transition_swiglu.py` | 95 | 16 | `args.mAuxOut.element_type.width == 16` |
| `transition/cute/gemm_transition_swiglu.py` | 197 | 128 | `K <= 128` |
| `transition/triton/fused.py` | 1092 | 512 | `K <= 512` |
| `transition/triton/fused.py` | 1467 | 128 | `K <= 128` |
| `transition/whole_op.py` | 41 | 256 | `d_hidden >= 256` |
| `transition/whole_op.py` | 49 | 256 | `d_hidden >= 256` |
| `triangle_attention/triton/atomic.py` | 630 | 32 | `D != 32` |
| `trimul_inproj/cute/back_split.py` | 36 | 128 | `N < 128` |
