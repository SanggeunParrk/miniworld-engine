# trimul_inproj — fused input projections for the triangle multiplicative update

A new fusion unit (not `tm1`/`tm2`): from the normalized pair `x` it produces
**all three** input projections in a single read of `x`.

## Math

```
left  = sigmoid(x @ WLg) * (x @ WL)     -> [B, D, L, L]   (feeds the bmm)
right = sigmoid(x @ WRg) * (x @ WR)     -> [B, D, L, L]   (feeds the bmm)
gate  = sigmoid(x @ Wg)                 -> [B, L, L, D]   (final elementwise mul)
```

`left`/`right` come out in `[B, D, L, L]` so the outgoing contraction is a flat
batched matmul (`einsum("bdik,bdjk->bdij")` == `torch.bmm`) with no transpose.
`gate` stays `[B, L, L, D]` because it multiplies the `[B, L, L, D]` final output
and never enters the bmm.

## Where it sits in the new trimul pipeline

```
1. LN_in(x)                         (fused_ln_mask / layernorm kernel)
2. trimul_inproj  -> left, right, gate     ← THIS KERNEL
3. tri = bmm(left, right)            (torch.bmm)
4. LN_out(tri) @ W_out  [+ gate ⊙]  (layernorm_linear kernel; fuse the mul in epilogue)
5. (mul folded into 4)
```

Pulling `gate` to the front (step 2) is what lets step 4 fold the final
`gate ⊙ ·` into the layernorm-linear epilogue, removing a full-tensor round trip
vs. materializing `out_normed` separately. See the memory-traffic analysis in the
project notes (12T vs the current cute path's 14T).

## Why this differs from `tm1`

`tm1` computes only `left`/`right`, as **two** `gemm_act(glu)` launches that each
re-read `x`. `trimul_inproj`:

- **fuses left+right into ONE launch** — stacks both sides' interleaved
  `[gate|proj]` weights into one `(D, 4D)` B operand, so `x` is read once and the
  glu epilogue emits `(M, 2D)`: cols `[:D]` = left, `[D:]` = right.
- **adds `gate`** to the same unit.

Tradeoff vs tm1's split: one wider GEMM (N=4D→2D postact) raises epilogue
register/accumulator pressure — tm1 split the sides specifically to keep that
low. Whether the single wider launch wins is a benchmark question (occupancy vs.
one fewer `x` read + one fewer launch).

## The `[B,D,L,L]` layout — two paths

**Fallback (default, `bdll_direct=False`)** — works on stock quack. Take the
natural n-major postact `[B, L, L, 2D]`, split into left/right, and
`permute(0,3,1,2).contiguous()` each to `[B, D, L, L]`. Correct, but pays the
transpose copy (the bottleneck this layout was meant to avoid).

**Direct (`bdll_direct=True`)** — the fast path. Pre-allocate `[B, 2D, L, L]` and
hand the GEMM an M-major *view* (shape `(M=L*L, N=2D)`, strides `(1, L*L)`) as the
postact; the kernel writes `(m, n)` at `m*1 + n*L*L`, straight into the planes, no
permute. `left = storage[:, :D]`, `right = storage[:, D:]` (contiguous for B=1).

The direct path needs the gated postact to be M-major, which **stock quack
rejects** (forces n-major: `gemm_act.py:203` assert + `:321` `pa_leading_dim = 1`).
The original `cute-env` patch for this was **not carried into the unified env**, so
we re-own it in-repo via `cute/_bdll_patch.py` — it re-derives the two patched
functions from quack's own source (one-line replacements) and applies them at
import. No quack file is modified; survives `pixi install`. (Note: a distinct
`__qualname__` on the patched compile fn busts quack's disk cache so a stale
n-major `.o` from a pre-patch run isn't reused.)

> 🔴 The same stock-quack gap also breaks **tm1's existing cute path**
> (`modules/triangle_multiplication/module.py::_forward_cute` →
> `tm1_cute_forward(..., out_layout="bdll_direct")`). `_bdll_patch.apply()` fixes
> that path too — just import+apply it there.

## Status

Correctness **verified** vs fp32 reference for **both** paths — `left`/`right`/
`gate` all OK for L ∈ {64,128,384,512}, bf16 + fp16, B=1 (`cute/verify.py`):
`bdll_direct=False` (permute fallback) and `bdll_direct=True` (in-repo patch).

## Current limitations (first cut)

- **`bdll_direct=True` is verified but not yet the default** (default `False` =
  permute fallback, no quack mutation). Switch to `True` for the layout perf win;
  benchmark the two against the current cute path before wiring.
- **`gate` is plain torch**, not fused: quack's non-gated `act_fn_map` has no
  `"sigmoid"`, and `gate = sigmoid(x@Wg)` has no proj partner for the glu
  epilogue. It's the only piece not in the gated launch. Fix paths (see
  `cute/launch.py` docstring): add `"sigmoid"` to `act_fn_map` (one extra fused
  gemm+sigmoid launch), or a custom `(D, 5D)` mixed-epilogue kernel (true single
  launch).
- **B = 1 only** (matches tm1's `bdll_direct` view/slice assumptions).

## Files

```
trimul_inproj/
├── __init__.py
├── reference.py        # pytorch reference (functional + nn.Module fwd/bwd)
├── interface.py        # trimul_inproj_cute(...) public entry (lazy cute import)
├── cute/
│   ├── launch.py       # the kernel: 1 gated GEMM (left+right, bdll) + torch gate
│   └── verify.py       # correctness vs fp32 reference
```
```

## Verify

```bash
srun --partition=h100 --account=cssb --qos=cssb_h100 --gres=gpu:h100:1 \
     --cpus-per-task=8 --mem=64G --time=00:20:00 \
     bash -c 'cd /home/psk6950/miniworld-kernels && PYTHONPATH=src \
     pixi run --frozen python \
     src/miniworld_kernels/kernels/trimul_inproj/cute/verify.py'
```
