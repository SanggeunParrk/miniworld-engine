"""Side-split TM1 forward backed by quack's SM90 gated GEMM.

For one side (e.g. left):

    a_full = x @ [Wg | Wp]                # (M, 2D), single fused GEMM
    out    = sigmoid(a_full[..., 0::2]) * a_full[..., 1::2]   # gated epilogue

Where ``[Wg | Wp]`` is interleaved in N so the epilogue picks (gate, proj)
pairs as adjacent columns. ``quack.gemm_act`` with ``activation="glu"`` does
exactly this on SM90 with a warp-specialized producer/consumer GEMM kernel.

Two side launches give us left and right; the user spec asked for this exact
split — each side launch holds only the gate + projection accumulators for
that side, not all four at once, so register pressure stays low.

Output layout: the user spec wants ``[B, d, L, L]`` directly. The fused
gated GEMM naturally produces ``[B*L*L, d]``; we reshape to ``[B, L, L, d]``
and ``permute(0, 3, 1, 2).contiguous()`` to match. The permute is an extra
read+write of the output but at d=128 it's a small fraction of the GEMM cost.
A future custom kernel can fold the transpose into the GEMM epilogue.
"""

from __future__ import annotations
from miniworld_engine.autotune.configs import configs_for

import torch
from miniworld_engine.kernels._quack_compat import gemm_act

import triton
import triton.language as tl

from miniworld_engine.autotune import tensor_dtype_of

# Flat elementwise glue around the cute GEMM: one tiled axis (the linear element index).
# Both kernels were launched with a literal BLOCK=1024 that no measurement chose.



from miniworld_engine.autotune.buckets import bucket_mixed as _bucket


def get_seq_group(rows) -> int:
    """Delegates to canonical size-bucketing (autotune.buckets)."""
    return _bucket(rows)


# restore_value is NOT optional here: this kernel is IN-PLACE (proj is both read and written),
# and the autotuner runs every candidate config on the real buffers. Without it, proj would be
# multiplied by sigmoid(gate) once per benched config instead of once, so the tuning sweep itself
# silently corrupts the tensor. Any in-place kernel that gains an @autotune needs this.
@triton.autotune(configs=configs_for("gated_projection_gate_inplace_flat_triton"), key=['shape_key'], restore_value=['proj_ptr'])
@triton.jit
def _gate_mul_kernel(proj_ptr, gate_ptr, n, BLOCK_E: tl.constexpr, shape_key):
    """In-place: proj = proj * sigmoid(gate), elementwise over a flat buffer.

    sigmoid + multiply are computed in fp32 (a single round to bf16 on store),
    replacing torch's separate ``sigmoid`` (temp alloc) + ``mul_`` — which made
    ~4 passes over the [B,D,L,L] tensor. This is one read of proj + one read of
    gate + one write. Precision is not reduced (fewer intermediate roundings).
    """
    pid = tl.program_id(0).to(tl.int64)
    offs = pid * BLOCK_E + tl.arange(0, BLOCK_E)
    m = offs < n
    p = tl.load(proj_ptr + offs, mask=m).to(tl.float32)
    g = tl.load(gate_ptr + offs, mask=m).to(tl.float32)
    tl.store(proj_ptr + offs, (p * tl.sigmoid(g)).to(tl.bfloat16), mask=m)




@triton.autotune(configs=configs_for("gated_projection_gate_packed_flat_triton"), key=['shape_key'])
@triton.jit
def _glu_wide_kernel(wide_ptr, out_ptr, MD, D_L2, BLOCK_E: tl.constexpr, shape_key):
    """out[flat < D_L2 region] = sigmoid(wide[gate half]) * wide[proj half].

    ``wide`` is (2D, L, L) contiguous == gate channels [0:D] then proj channels
    [D:2D], each (L,L). Flattened, gate elem e maps to proj elem e + D_L2. One read
    of gate + one of proj + one write, fp32 math, single bf16 round on store."""
    pid = tl.program_id(0).to(tl.int64)
    offs = pid * BLOCK_E + tl.arange(0, BLOCK_E)
    m = offs < D_L2
    g = tl.load(wide_ptr + offs, mask=m).to(tl.float32)              # gate half
    p = tl.load(wide_ptr + offs + D_L2, mask=m).to(tl.float32)       # proj half
    tl.store(out_ptr + offs, (p * tl.sigmoid(g)).to(tl.bfloat16), mask=m)


def _glu_wide(out: torch.Tensor, wide: torch.Tensor, D: int, L: int) -> None:
    """out[B,D,L,L] = sigmoid(wide[:,0:D]) * wide[:,D:2D], wide is (B,2D,L,L). B=1."""
    D_L2 = D * L * L
    grid = lambda meta: (triton.cdiv(D_L2, meta["BLOCK_E"]),)  # noqa: E731
    _glu_wide_kernel[grid](wide, out, 2 * D_L2, D_L2, shape_key=get_seq_group(D_L2))


def _fused_gate_mul(proj: torch.Tensor, gate: torch.Tensor) -> None:
    """proj *= sigmoid(gate), fused single-pass Triton kernel. In-place on proj."""
    assert proj.is_contiguous() and gate.is_contiguous()
    n = proj.numel()
    grid = lambda meta: (triton.cdiv(n, meta["BLOCK_E"]),)  # noqa: E731
    _gate_mul_kernel[grid](proj, gate, n, shape_key=get_seq_group(n))



def tm1_cute_forward(
    x: torch.Tensor,  # (B, L, L, D), contiguous, bf16/fp16
    WL: torch.Tensor,  # (D, D)  — to_left.weight.T
    WLg: torch.Tensor,  # (D, D)  — to_left_gate.weight.T
    WR: torch.Tensor,  # (D, D)  — to_right.weight.T
    WRg: torch.Tensor,  # (D, D)  — to_right_gate.weight.T
    *,
    out_layout: str = "bdll",  # "bdll" (spec) or "blld" (cheap path, no permute)
) -> tuple[torch.Tensor, torch.Tensor]:
    """Forward TM1 (left+right gated dual GEMM). Returns ``(left, right)`` in
    ``[B, d, L, L]`` contiguous layout (different from the original
    ``[B, L, L, d]`` Triton convention — see module docstring).
    """
    assert x.dim() == 4, f"expected (B, L, L, D), got {tuple(x.shape)}"
    assert x.is_cuda and x.is_contiguous(), "x must be cuda + contiguous"
    B, L1, L2, D = x.shape
    assert L1 == L2, f"expected square pair, got L1={L1} L2={L2}"
    L = L1
    for W, name in ((WL, "WL"), (WLg, "WLg"), (WR, "WR"), (WRg, "WRg")):
        assert W.shape == (D, D), f"{name}: expected ({D},{D}), got {tuple(W.shape)}"
        assert W.dtype == x.dtype, f"{name}: dtype mismatch ({W.dtype} vs {x.dtype})"

    M = B * L * L
    x_flat = x.reshape(M, D)

    # gemm_act's "glu" epilogue computes ``sigmoid(D[:, 2i]) * D[:, 2i+1]``.
    # The B operand must therefore be **pre-interleaved**: gate columns at even
    # positions, projection at odd. (We probed: ``concat_layout=("B",)`` on this
    # quack release does not produce the right interleaving for our B layout.)
    def _interleave(Wg: torch.Tensor, Wp: torch.Tensor) -> torch.Tensor:
        out = torch.empty(D, 2 * D, device=Wg.device, dtype=Wg.dtype)
        out[:, 0::2] = Wg
        out[:, 1::2] = Wp
        return out

    B_left = _interleave(WLg, WL)  # (D, 2D), gate at even cols, proj at odd
    B_right = _interleave(WRg, WR)

    if out_layout in ("blld", "bdll"):
        _, left_flat = gemm_act(A=x_flat, B=B_left, activation="glu", store_preact=False)
        _, right_flat = gemm_act(A=x_flat, B=B_right, activation="glu", store_preact=False)
        if out_layout == "blld":
            return left_flat.view(B, L, L, D), right_flat.view(B, L, L, D)
        # NOTE: permute().contiguous() here is the bottleneck — at L=1024 it
        # takes ~4.3ms per side vs ~0.2ms for the GEMM itself. The
        # ``bdll_direct`` path below skips it entirely by handing the kernel
        # an M-major view of pre-allocated [B,D,L,L] storage.
        left_bdll = left_flat.view(B, L, L, D).permute(0, 3, 1, 2).contiguous()
        right_bdll = right_flat.view(B, L, L, D).permute(0, 3, 1, 2).contiguous()
        return left_bdll, right_bdll
    if out_layout == "bdll_direct_wide":
        # Like bdll_direct but ONE wide GEMM per side (N=2D, [gate|proj] concatenated) so
        # A (=x) is loaded ONCE per side instead of twice, and there is one GEMM launch per
        # side instead of two. act=None M-major store (no glu epilogue -> no stride-8 issue),
        # then a triton kernel folds sigmoid(gate)*proj into [B,D,L,L]. Aims to recover the
        # hand-rolled collective's shared-A-load / fewer-launch win via quack primitives.
        if B != 1:
            raise NotImplementedError("bdll_direct_wide currently only supports B=1")
        Bcat_L = torch.cat([WLg, WL], dim=1)  # (D, 2D): gate cols [0:D], proj cols [D:2D]
        Bcat_R = torch.cat([WRg, WR], dim=1)

        def _side(Bcat: torch.Tensor) -> torch.Tensor:
            wide = torch.empty(B, 2 * D, L, L, device=x.device, dtype=x.dtype)
            wide_view = wide.view(2 * D, L * L).T  # (M=L*L, 2D) M-major
            gemm_act(A=x_flat, B=Bcat, activation=None, store_preact=False, postact_out=wide_view)
            out = torch.empty(B, D, L, L, device=x.device, dtype=x.dtype)
            _glu_wide(out, wide, D, L)
            return out

        return _side(Bcat_L), _side(Bcat_R)
    if out_layout == "bdll_glu":
        # Fused gated GEMM (quack gemm_act "glu") storing the GLU result M-major straight
        # into [B,D,L,L] (zero-copy, NO permute) — the quack-0.5.0 equivalent of the old
        # hand-rolled bdll_sm100 collective: one A load, dual-B (interleaved [gate|proj]),
        # fused sigmoid-gate epilogue. Recovers the fusion bdll_direct dropped (2 plain
        # GEMMs + a separate triton gate) while keeping the zero-copy M-major store.
        if B != 1:
            raise NotImplementedError("bdll_glu currently only supports B=1")

        def _bdll_buf():
            t = torch.empty(B, D, L, L, device=x.device, dtype=x.dtype)
            return t, t.view(D, L * L).T  # (storage[B,D,L,L], M-major (M=L*L,N=D) view)

        left_bdll, left_view = _bdll_buf()
        right_bdll, right_view = _bdll_buf()
        # glu halves N: B is (D, 2D) interleaved -> postact is (M, D) = sigmoid(gate)*proj.
        gemm_act(A=x_flat, B=B_left, activation="glu", store_preact=False, postact_out=left_view)
        gemm_act(A=x_flat, B=B_right, activation="glu", store_preact=False, postact_out=right_view)
        return left_bdll.view(B, D, L, L), right_bdll.view(B, D, L, L)
    if out_layout == "bdll_direct":
        # Zero-copy [B, D, L, L] store with NO transpose: pre-allocate the
        # [B, D, L, L] storage and hand the GEMM an M-major view of it
        # (shape (M=L*L, N=D), strides (1, L*L)) as postact_out. Row index
        # m == flat-(r,c) lands at stride 1, column n == D at stride L*L, i.e.
        # the kernel writes straight into [B,D,L,L] — killing the ~4.5ms
        # permute of the "bdll" path.
        #
        # Why NOT the fused "glu" gated GEMM here: quack 0.3.11's *gated*
        # epilogue scrambles an M-major postact (its smem tile stays n-major
        # while TMA stores m-major). Its *non-gated* store is correct for
        # M-major, so we split the side into two plain GEMMs (proj = x@Wp,
        # gate = x@Wg) straight into M-major [B,D,L,L] views and fuse
        # sigmoid(gate)*proj pointwise. Same GEMM FLOPs as the gated path;
        # the only extra cost is one gate write + a cheap pointwise.
        if B != 1:
            raise NotImplementedError("bdll_direct currently only supports B=1")

        def _bdll_buf():
            t = torch.empty(B, D, L, L, device=x.device, dtype=x.dtype)
            return t, t.view(D, L * L).T  # (storage[B,D,L,L], M-major (M=L*L,N=D) view)

        left_bdll, left_view = _bdll_buf()
        right_bdll, right_view = _bdll_buf()
        gate_buf, gate_view = _bdll_buf()  # reused for both sides

        # left = sigmoid(x @ WLg) * (x @ WL)
        gemm_act(A=x_flat, B=WL, activation=None, store_preact=False, postact_out=left_view)
        gemm_act(A=x_flat, B=WLg, activation=None, store_preact=False, postact_out=gate_view)
        _fused_gate_mul(left_bdll, gate_buf)

        # right = sigmoid(x @ WRg) * (x @ WR)
        gemm_act(A=x_flat, B=WR, activation=None, store_preact=False, postact_out=right_view)
        gemm_act(A=x_flat, B=WRg, activation=None, store_preact=False, postact_out=gate_view)
        _fused_gate_mul(right_bdll, gate_buf)
        return left_bdll, right_bdll
    if out_layout == "bdll_sm100":
        # From-scratch SM100 tcgen05 gated GEMM: out = sigmoid(x@Wg.T) * (x@Wp.T),
        # stored M-major straight into [B, D, L, L] (zero-copy) -- replaces the two
        # non-gated quack GEMMs + the Triton gate kernel AND the permute in one launch/side.
        if B != 1:
            raise NotImplementedError("bdll_sm100 currently only supports B=1")
        # Built on CUTLASS's tuned Blackwell persistent collective (dual-B fused GLU):
        # K-tiled mainloop, N-tiled by the persistent scheduler, dual-TMEM accumulator
        # guarded to <=512 cols; handles the full d range. (The earlier from-scratch
        # `naive` gate GEMM — full-K/N single buffer, d=128-only, strictly slower — was
        # removed; the collective supersedes it.)
        from miniworld_engine.kernels.tm1.cute.sm100_gate_gemm_collective import gate_gemm as _sm100_gate_gemm

        # gate_gemm computes A@Bp.T ; tm1 wants x@W.T with W = to_*.weight, and the
        # WL/WLg/... passed here are W.T, so Bp = (W.T).mT = W (contiguous).
        BLp = WL.mT.contiguous()
        BLg = WLg.mT.contiguous()
        BRp = WR.mT.contiguous()
        BRg = WRg.mT.contiguous()
        left_dll = _sm100_gate_gemm(x_flat, BLp, BLg, mmajor=True)   # (D, L*L) == [B,D,L,L]
        right_dll = _sm100_gate_gemm(x_flat, BRp, BRg, mmajor=True)
        return left_dll.view(B, D, L, L), right_dll.view(B, D, L, L)
    raise ValueError(f"unknown out_layout: {out_layout!r}")
