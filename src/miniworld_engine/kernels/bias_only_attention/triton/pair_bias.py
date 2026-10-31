"""The pair bias of the bias-only attention in ONE pass over the pair: ``logits^T[h, r] = (LN0(pair)[r] @ W'^T)[h]``, forward and backward.

``BiasOnlyAttention`` computes ``to_bias(ln_pair(pair))`` where ``ln_pair`` has no offset, so with ``LN0`` the LayerNorm without affine and
``W' = W diag(gamma)`` (the fold ``modules/bias_only_dit/hoist.py`` already uses) the bias is ``LN0(pair) @ W'^T``: a LayerNorm over the
``C = d_pair`` channels of every pair entry, then a ``C -> H`` projection. Unfused that is a LayerNorm that reads and writes the whole pair
(the largest tensor of the block: ``B L L C``), a GEMM that reads the normalised pair again, and in the backward a GEMM that writes a pair
sized gradient, the LayerNorm backward that reads two and writes one, and a skinny weight-gradient GEMM over all ``B L L`` rows: ~ 270 us
of a 3.2 ms bias-only DiT training step at L = 512, and all of it is memory traffic. The projection is tiny (``H <= 32`` outputs), so both
directions fit in one pass that holds a row block in registers:

    forward   x[BR, C] -> mean, rstd -> xhat -> tl.dot(W', xhat^T) -> logits^T[H, BR]       reads the pair once, writes ``H`` values per row
    backward  dxhat = dlogits^T @ W' ; dx = rstd (dxhat - mean(dxhat) - xhat mean(dxhat xhat)) ; dW' += dlogits @ xhat
              reads the pair and ``H`` values per row, writes the pair gradient; dW' is accumulated in registers by a persistent program
              and added to the fp32 result once per program.

The logits come out HEAD-MAJOR (``[H, B*L*L]``, the layout the softmax over the keys wants), so the ``permute(0, 3, 1, 2)`` of the unfused form
is a view. Registered as ``bias_only_attention_layernorm_gemm_triton`` (forward) and ``bias_only_attention_bwd_layernorm_gemm_triton``
(backward). Rows per program (``BR``), warps and stages are the autotuned axes; the backward's persistent grid is sized from the ``BR`` a
config carries and the program count is read in-kernel. The default config set is the launch these kernels used before they were
registered (64 rows per program, swept once on an H100, ``docs/records/portable_bodit_ct_training_20261026.md``; forward 4 warps; backward
4 warps, 8 from 256 channels on, ``_bwd_warps`` applied by the early prune outside a build), so a card with no tuned cache runs exactly
what it ran.
"""

from __future__ import annotations

import functools

import torch
import triton
import triton.language as tl

from miniworld_engine.autotune.configs import configs_for
from miniworld_engine.autotune.shape_key import both_key
from miniworld_engine.kernels._compile import opaque
from miniworld_engine.kernels._device import num_sms
from miniworld_engine.kernels._tf32 import tf32_dot




def _bwd_warps(c: int) -> int:
    """Warps of the backward program: 4, or 8 from 256 channels on -- its [64, C] row block, d xhat and the [16, C] dW' accumulator spill
    at 4 warps there (H100, R = 4 x 512^2, 16 outputs, 2026-10-29: C 256 bf16 1611 -> 582 us, fp32 2588 -> 1040 us; C 128 stays at 4: 249
    against 395 us)."""
    return 8 if c >= 256 else 4


def _bwd_prune(configs, nargs, **meta):
    """Outside a build, the warps ``_bwd_warps`` picks for this channel count (the launch these kernels had before they were tuned);
    a build (``settings.run_autotune``) searches every config."""
    from miniworld_engine import settings

    if settings.current().run_autotune:
        return configs
    want = _bwd_warps(int(nargs["C"] if "C" in nargs else meta["C"]))   # C reaches the prune as a launch keyword
    keep = [c for c in configs if c.num_warps == want]
    return keep or configs


@triton.autotune(configs=configs_for("bias_only_attention_layernorm_gemm_triton"), key=["shape_key"])
@triton.jit
def _ln_bias_fwd_kernel(X, W, OUT, MEAN, RSTD, R, eps, shape_key, C: tl.constexpr, N: tl.constexpr, NP: tl.constexpr, BR: tl.constexpr):
    pid = tl.program_id(0).to(tl.int64)
    rows = pid * BR + tl.arange(0, BR)
    rm = rows < R
    cols = tl.arange(0, C)
    x = tl.load(X + rows[:, None] * C + cols[None, :], mask=rm[:, None], other=0.0).to(tl.float32)
    mean = tl.sum(x, axis=1) / C
    xc = x - mean[:, None]
    rstd = tl.rsqrt(tl.sum(xc * xc, axis=1) / C + eps)
    xh = (xc * rstd[:, None]).to(X.dtype.element_ty)
    hh = tl.arange(0, NP)
    w = tl.load(W + hh[:, None] * C + cols[None, :], mask=(hh < N)[:, None], other=0.0)
    acc = tf32_dot(w, tl.trans(xh))                                            # [NP, BR]; fp32: TF32 rounded to nearest
    tl.store(OUT + hh[:, None] * R + rows[None, :], acc.to(OUT.dtype.element_ty), mask=(hh < N)[:, None] & rm[None, :])
    tl.store(MEAN + rows, mean, mask=rm)
    tl.store(RSTD + rows, rstd, mask=rm)


@triton.autotune(configs=configs_for("bias_only_attention_bwd_layernorm_gemm_triton"), key=["shape_key"],
                 prune_configs_by={"early_config_prune": _bwd_prune}, reset_to_zero=["DW"])
@triton.jit
def _ln_bias_bwd_kernel(X, W, DL, MEAN, RSTD, DX, DW, R, shape_key, C: tl.constexpr, N: tl.constexpr, NP: tl.constexpr, BR: tl.constexpr):
    pid = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, C)
    hh = tl.arange(0, NP)
    hm = hh < N
    w = tl.load(W + hh[:, None] * C + cols[None, :], mask=hm[:, None], other=0.0)
    dw = tl.zeros((NP, C), dtype=tl.float32)
    for b in range(pid, tl.cdiv(R, BR), tl.num_programs(0)):
        rows = b * BR + tl.arange(0, BR)
        rm = rows < R
        x = tl.load(X + rows[:, None] * C + cols[None, :], mask=rm[:, None], other=0.0).to(tl.float32)
        mean = tl.load(MEAN + rows, mask=rm, other=0.0)
        rstd = tl.load(RSTD + rows, mask=rm, other=0.0)
        xh = (x - mean[:, None]) * rstd[:, None]
        dl = tl.load(DL + hh[:, None] * R + rows[None, :], mask=hm[:, None] & rm[None, :], other=0.0)   # [NP, BR]
        dxh = tf32_dot(tl.trans(dl), w)                                                                  # [BR, C]
        dw = tf32_dot(dl, xh.to(X.dtype.element_ty), dw)
        c1 = tl.sum(dxh, axis=1) / C
        c2 = tl.sum(dxh * xh, axis=1) / C
        dx = rstd[:, None] * (dxh - c1[:, None] - xh * c2[:, None])
        tl.store(DX + rows[:, None] * C + cols[None, :], dx.to(DX.dtype.element_ty), mask=rm[:, None])
    tl.atomic_add(DW + hh[:, None] * C + cols[None, :], dw, mask=hm[:, None])


def _np(n: int) -> int:
    return max(16, triton.next_power_of_2(n))


def supported(pair: torch.Tensor, n_out: int) -> bool:
    """A power-of-two channel count (the row block holds a whole row) and at most 64 outputs, bf16 or fp32, contiguous."""
    c = pair.shape[-1]
    return (pair.is_cuda and pair.dtype in (torch.bfloat16, torch.float32) and c == triton.next_power_of_2(c) and 16 <= c <= 512
            and 1 <= n_out <= 64 and pair.is_contiguous())


def _fwd_fake(x, w, eps=1e-5):
    """(logits^T [N, R], mean [R], rstd [R])."""
    return x.new_empty((w.shape[0], x.shape[0])), x.new_empty(x.shape[0], dtype=torch.float32), x.new_empty(x.shape[0], dtype=torch.float32)


@opaque(fake=_fwd_fake, name="bias_only_attention_pair_bias_fwd")
def _fwd(x: torch.Tensor, w: torch.Tensor, eps: float = 1e-5) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """x [R, C], w [N, C] (the folded weights, x's dtype) -> (logits^T [N, R], mean [R], rstd [R])."""
    r, c = x.shape
    n = w.shape[0]
    out = torch.empty((n, r), device=x.device, dtype=x.dtype)
    mean = torch.empty(r, device=x.device, dtype=torch.float32)
    rstd = torch.empty_like(mean)
    _ln_bias_fwd_kernel[lambda meta: (triton.cdiv(r, meta["BR"]),)](x, w, out, mean, rstd, r, eps, shape_key=both_key(r, C=c, N=n), C=c,
                                                                    N=n, NP=_np(n))
    return out, mean, rstd


def _bwd_fake(x, w, dl, mean, rstd):
    """(dx like ``x``, dW' [N, C] fp32)."""
    return torch.empty_like(x), x.new_empty(w.shape, dtype=torch.float32)


@opaque(fake=_bwd_fake, name="bias_only_attention_pair_bias_bwd")
def _bwd(x: torch.Tensor, w: torch.Tensor, dl: torch.Tensor, mean: torch.Tensor, rstd: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """-> (dx [R, C] in x's dtype, dW' [N, C] fp32)."""
    r, c = x.shape
    n = w.shape[0]
    dx = torch.empty_like(x)
    dw = torch.zeros((n, c), device=x.device, dtype=torch.float32)
    sms = num_sms(x.device.index or 0)
    # persistent: at most two programs per SM, each looping over row blocks of the config's BR
    _ln_bias_bwd_kernel[lambda meta: (max(1, min(sms * 2, triton.cdiv(r, meta["BR"]))),)](
        x, w, dl, mean, rstd, dx, dw, r, shape_key=both_key(r, C=c, N=n), C=c, N=n, NP=_np(n))
    return dx, dw


class PairBiasFn(torch.autograd.Function):
    """``logits^T = (LN0(pair) @ (W diag(gamma))^T)^T`` as [H, B*L*L], differentiable in the pair, ``gamma`` (the LayerNorm weight) and ``W``."""

    @staticmethod
    def forward(ctx, pair, gamma, weight, eps):
        c = pair.shape[-1]
        x = pair.reshape(-1, c)
        wf = (weight.float() * gamma.float()[None, :]).to(x.dtype)       # the fold; fp32 parameters, rounded once to the activation dtype
        out, mean, rstd = _fwd(x, wf, eps)
        ctx.save_for_backward(x, wf, mean, rstd, gamma, weight)
        ctx.shape = pair.shape
        return out

    @staticmethod
    def backward(ctx, dl):
        x, wf, mean, rstd, gamma, weight = ctx.saved_tensors
        dl = dl.to(x.dtype).contiguous()
        dx, dwf = _bwd(x, wf, dl, mean, rstd)
        dgamma = (dwf * weight.float()).sum(0).to(gamma.dtype)
        dweight = (dwf * gamma.float()[None, :]).to(weight.dtype)
        return dx.reshape(ctx.shape), dgamma, dweight, None


def pair_bias_logits(pair: torch.Tensor, gamma: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """``to_bias(ln_pair(pair))`` as head-major logits ``[B, H, L, L]`` for ``pair`` [B, L, L, C] (a view of the kernel's [H, B*L*L] output)."""
    b, l1, l2, _ = pair.shape
    out = PairBiasFn.apply(pair, gamma, weight, eps)
    return out.view(weight.shape[0], b, l1, l2).permute(1, 0, 2, 3)
