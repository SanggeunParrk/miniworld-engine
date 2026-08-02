"""Fused trimul input projection on quack's SM90 gated GEMM.

Computes, in ONE read of the normalized pair ``x``:

    left  = sigmoid(x @ WLg) * (x @ WL)     -> [B, D, L, L]
    right = sigmoid(x @ WRg) * (x @ WR)     -> [B, D, L, L]
    gate  = sigmoid(x @ Wg)                 -> [B, L, L, D]

HOW left+right are fused (vs ``tm1``'s two launches):
  tm1 runs two separate ``gemm_act(glu)`` launches, each re-reading ``x``. Here
  we stack both sides' interleaved ``[gate|proj]`` weights into ONE ``(D, 4D)``
  B operand, so a single GEMM reads ``x`` once and the glu epilogue emits a
  ``(M, 2D)`` postact: columns ``[:D]`` = left, ``[D:]`` = right.

THE [B,D,L,L] DIRECT-WRITE (same trick as tm1's ``bdll_direct``):
  We pre-allocate combined storage ``[B, 2D, L, L]`` and hand the kernel an
  M-major *view* of it — shape ``(M=L*L, N=2D)``, strides ``(1, L*L)`` — as the
  postact output. The GEMM writes logical ``(m, n)`` at ``m*1 + n*L*L``, i.e.
  straight into the ``[2D, L, L]`` planes, no permute. ``left``/``right`` are
  then the first/second ``D`` planes (contiguous slices for B=1).
  Requires the patched quack (n-major postact assert relaxed) — already in this
  env, since the tm1 cute path uses it.

THE GATE caveat:
  ``gate = sigmoid(x @ Wg)`` has no projection partner, so it can't share the
  glu epilogue, and quack's non-gated ``act_fn_map`` has no ``"sigmoid"`` entry.
  For this first, verifiable cut we compute it in plain torch. It is the only
  part NOT fused into the gated launch. Two ways to fuse it later:
    1. add ``"sigmoid"`` to quack ``act_fn_map`` and run a second (non-gated)
       ``gemm_act(activation="sigmoid")`` — one fused gemm+sigmoid launch.
    2. fork ``GemmGatedSm90`` with a custom epilogue that does glu on the first
       4D cols and plain sigmoid on a trailing D cols of a single (D, 5D) GEMM
       — true single launch, x read once for all three. (See
       ``kernels/tm1/cute/gated_gemm_readable.py``.)

B=1 only (matches tm1's bdll_direct). Run on a COMPUTE NODE (srun), never login.
"""

from __future__ import annotations

import torch
from miniworld_engine.kernels._quack_compat import gemm_act


def _interleave(Wg: torch.Tensor, Wp: torch.Tensor) -> torch.Tensor:
    """(D,D),(D,D) -> (D,2D) with gate at even cols, proj at odd.

    glu epilogue computes ``sigmoid(D[:, 2i]) * D[:, 2i+1]``, so the B operand
    must carry (gate, proj) as adjacent columns. Same as ``tm1`` ``_interleave``.
    """
    D = Wg.shape[1]
    out = torch.empty(Wg.shape[0], 2 * D, device=Wg.device, dtype=Wg.dtype)
    out[:, 0::2] = Wg
    out[:, 1::2] = Wp
    return out


def prepack_lr_operand(
    WL: torch.Tensor,   # (D, D)  — to_left.weight.T
    WLg: torch.Tensor,  # (D, D)  — to_left_gate.weight.T
    WR: torch.Tensor,   # (D, D)  — to_right.weight.T
    WRg: torch.Tensor,  # (D, D)  — to_right_gate.weight.T
) -> torch.Tensor:
    """Build the fused ``(D, 4D)`` GLU B-operand ONCE (no per-forward cat/interleave).

    Columns: ``[interleave(WLg,WL) | interleave(WRg,WR)]`` — gate/proj interleaved
    per side, left side then right side. Pass the result to
    ``trimul_inproj_cute_forward(..., b_lr=...)``. Rebuild only when the weights
    change (e.g. once per optimizer step in training; once ever for inference) —
    not every forward.
    """
    return torch.cat([_interleave(WLg, WL), _interleave(WRg, WR)], dim=1)


def trimul_inproj_cute_forward(
    x: torch.Tensor,  # 4D pair: (B, L, L, D) if hidden_dim=-1/3, (B, D, L, L) if hidden_dim=1
    WL: torch.Tensor,  # (D, D)  — to_left.weight.T
    WLg: torch.Tensor,  # (D, D)  — to_left_gate.weight.T
    WR: torch.Tensor,  # (D, D)  — to_right.weight.T
    WRg: torch.Tensor,  # (D, D)  — to_right_gate.weight.T
    Wg: torch.Tensor | None,  # (D, D)  — to_gate.weight.T; None to skip gate
    *,
    bdll_direct: bool = False,
    compute_gate: bool = True,
    b_lr: torch.Tensor | None = None,  # pre-packed (D, 4H); skips cat/interleave
    hidden_dim: int = -1,  # where D lives in x: -1/3 -> BLLD, 1 -> BDLL
    out_hidden: int | None = None,  # per-side output width H (left/right each H wide); None -> D
    return_preact: bool = False,  # also return the pre-glu activation [B,4H,L,L] (for front bwd)
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Forward. Returns ``(left_bdll, right_bdll, gate_blld)``.

    ``left``/``right`` are ``[B, D, L, L]`` contiguous; ``gate`` is
    ``[B, L, L, D]`` contiguous (or ``None`` if ``compute_gate=False``).

    ``compute_gate=False`` returns only the fused left+right (gate is ``None``):
    use this when the back half (e.g. the cute ``tm2`` dual-gemm) recomputes the
    gate itself — materializing the gate here then costs an extra HBM round trip.

    ``bdll_direct``
        If True, write left/right straight into ``[B, 2D, L, L]`` via an M-major
        postact view (no permute) — the fast path. REQUIRES patched quack
        (n-major postact assert relaxed + ``pa_leading_dim`` follows the detected
        layout). Stock quack forces n-major postact for the gated epilogue and
        raises a runtime stride error. Default False uses the n-major postact +
        permute fallback, which works on stock quack but pays the transpose copy.

    ``hidden_dim``
        Position of the channel (D) axis in ``x``, so the same kernel can read the
        pair in either layout without a pre-permute:
          ``-1`` / ``3`` -> ``x`` is ``[B, L, L, D]`` (BLLD, channel-last; default).
          ``1``          -> ``x`` is ``[B, D, L, L]`` (BDLL, channel-first), e.g. handed
                            straight from a prior bmm stage. The GEMM A operand is then an
                            M-major view (channel strided by L*L) — no transpose copy.
        Output layout is unaffected (left/right are ``[B, D, L, L]``, gate ``[B, L, L, D]``).
    """
    assert x.dim() == 4, f"expected a 4D pair tensor, got {tuple(x.shape)}"
    assert x.is_cuda and x.is_contiguous(), "x must be cuda + contiguous"
    # hidden_dim names where the channel (D) axis lives in x:
    #   -1 / 3 -> BLLD (B, L, L, D), the channel-last default
    #    1     -> BDLL (B, D, L, L), channel-first (e.g. straight from a prior bmm stage)
    hd = hidden_dim % x.dim()
    if hd == 3:  # BLLD
        B, L1, L2, D = x.shape
    elif hd == 1:  # BDLL
        B, D, L1, L2 = x.shape
    else:
        raise ValueError(f"hidden_dim must select dim 1 (BDLL) or -1/3 (BLLD), got {hidden_dim}")
    assert L1 == L2, f"expected square pair, got L1={L1} L2={L2}"
    if B != 1:
        raise NotImplementedError("trimul_inproj cute path currently supports B=1")
    L = L1
    # H = per-side output width (left/right each H wide). Default H=D (square trimul);
    # bidirectional passes out_hidden=2*d_hidden so each side carries [outgoing|incoming].
    H = out_hidden if out_hidden is not None else D
    weights = [] if b_lr is not None else [(WL, "WL"), (WLg, "WLg"), (WR, "WR"), (WRg, "WRg")]
    if compute_gate:
        weights.append((Wg, "Wg"))
    for W, name in weights:
        exp = (D, D) if name == "Wg" else (D, H)  # gate is always D->D; proj weights D->H
        assert W.shape == exp, f"{name}: expected {exp}, got {tuple(W.shape)}"
        assert W.dtype == x.dtype, f"{name}: dtype mismatch ({W.dtype} vs {x.dtype})"

    M = B * L * L
    # A operand (M=L*L, K=D). BLLD reshapes to a contiguous k-major (M, D). BDLL has the
    # channel axis strided by L*L, so the load is an M-major (transposed) view — quack's
    # gated GEMM accepts m-major A directly (verified bit-identical to the k-major path).
    if hd == 3:
        x_flat = x.reshape(M, D)
    else:
        x_flat = x.reshape(B, D, M)[0].t()  # (M, D), strides (1, L*L); B=1

    # --- left + right: ONE gated GEMM, reads x once ---------------------------
    # B operand (D, 4D): [interleave(WLg,WL) | interleave(WRg,WR)], glu output
    # (M, 2D): cols [:D] = left, [D:] = right. Prefer a pre-packed `b_lr` (built
    # once via prepack_lr_operand) to skip the per-forward cat/interleave.
    if b_lr is None:
        b_lr = torch.cat([_interleave(WLg, WL), _interleave(WRg, WR)], dim=1)  # (D, 4H)
    else:
        assert b_lr.shape == (D, 4 * H), f"b_lr: expected ({D},{4*H}), got {tuple(b_lr.shape)}"
    B_lr = b_lr

    if bdll_direct:
        # FAST path: M-major view of [B, 2H, L, L]. Stock quack rejects an
        # M-major gated postact; our in-repo shim owns that policy (no quack file
        # is modified). See _bdll_patch.py.
        try:
            from . import _bdll_patch
        except ImportError:  # script / sys.path context (e.g. verify.py)
            import _bdll_patch

        _bdll_patch.apply()
        lr = torch.empty(B, 2 * H, L, L, device=x.device, dtype=x.dtype)
        lr_view = lr.view(2 * H, L * L).T  # (M, 2H) strides (1, L*L)
        # preact written M-major straight into a [B,4H,L,L] buffer (same no-transpose
        # trick as the postact) — channels = b_lr column order, exactly what
        # front_bwd_fused wants. Avoids a (M,4H)->(4H,M) transpose in the forward.
        preact_buf = torch.empty(B, 4 * H, L, L, device=x.device, dtype=x.dtype) \
            if return_preact else None
        preact_view = preact_buf.view(4 * H, L * L).T if return_preact else None  # (M,4H) str(1,M)
        gemm_act(
            A=x_flat, B=B_lr, activation="glu", store_preact=return_preact,
            preact_out=preact_view, postact_out=lr_view,
        )
        left_bdll = lr[:, :H]  # (B, H, L, L), contiguous for B=1
        right_bdll = lr[:, H:]  # (B, H, L, L), contiguous for B=1
        if return_preact:
            return left_bdll, right_bdll, preact_buf
    else:
        # FALLBACK (stock quack): natural n-major postact [B,L,L,2H], then permute.
        _, lr_flat = gemm_act(A=x_flat, B=B_lr, activation="glu", store_preact=False)
        lr_blld = lr_flat.view(B, L, L, 2 * H)  # (B, L, L, 2H)
        left_bdll = lr_blld[..., :H].permute(0, 3, 1, 2).contiguous()  # (B,H,L,L)
        right_bdll = lr_blld[..., H:].permute(0, 3, 1, 2).contiguous()  # (B,H,L,L)

    # --- gate: fused gemm + sigmoid in ONE quack launch (not torch) -----------
    # `gemm_act(activation="sigmoid")` = sigmoid(x @ Wg) with the sigmoid fused in
    # the epilogue (registers), so no separate matmul-write + sigmoid-read-write.
    # Needs "sigmoid" registered in quack's act_fn_map (our shim). gate stays blld
    # (M, D) -> [B, L, L, D] for the final elementwise mul.
    gate = None
    if compute_gate:
        try:
            from . import _bdll_patch
        except ImportError:  # script / sys.path context
            import _bdll_patch

        _bdll_patch.ensure_sigmoid_act()
        _, gate_flat = gemm_act(A=x_flat, B=Wg, activation="sigmoid", store_preact=False)
        gate = gate_flat.view(B, L, L, D)  # (B, L, L, D)

    return left_bdll, right_bdll, gate
