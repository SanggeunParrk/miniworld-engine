"""MSAPairWeightedAveraging on sm_90a for the shapes the MiniWorld kernels (`pwa_train.py`) do not cover: d_msa 64 / 128,
d_pair 128 / 256 / 384, 8 heads x 8 / 16 value channels (Protenix-v2, ESMFold2, OpenDDE), any MSA depth, any length up to
1024 (a length that is not a multiple of 16 is padded with masked tokens), inference and training, bf16.

The structure is the B200 one (`csrc/sm100/pwa_sm100.cu`), written for wgmma instead of tcgen05:

Forward   pair_fwd3 (csrc/pair_wide.cu: LayerNorm(d_pair) -> proj_z -> key mask -> softmax over j, mma.sync, d_pair a compile-time
                     width)  -> w [H, N, N]
          ln_vg_w   (csrc/pwa_wide.cu: y = LN(m), the LN statistics, v = y Wv^T head-major [H, N, S*C])
          cuBLAS    o = w . v
          gate_out_w (gate = y Wg^T, o * sigmoid(gate) straight from registers into the out projection, dropout, residual)
Backward  gate_bwd_w (do' = dout Wo, the gate recomputed from y: do (head-major), go, dgp)
          cuBLAS    dv = w^T . do, dw = do . v^T, and the weight gradients dWo = dout^T go, dWg | dWv = [dgp | dv]^T y
          dm_bwd_w  (dy = [dgp | dv] [Wg; Wv], LayerNorm backward + the residual gradient -> dm, dgamma, dbeta)
          pair_bwd3 (softmax backward -> proj_z backward -> LayerNorm backward -> dz, dWb, dgamma_z, dbeta_z)

Every kernel is built per (d_msa, d_hidden) / d_pair. The MiniWorld shape (64 / 128 / 8 x 32) stays on `pwa_train.py`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from miniworld_engine.kernels._compile import opaque

H = 8
DMSA = (64, 128)
DPAIR = (128, 256, 384)
DHID = (8, 16)
# (d_msa, d_pair, d_hidden) served by the BIG build of `pwa_wide.cu` (8 heads x 32 = 256 hidden channels, IntelliFold2): the gate /
# out pass of `gate_out_w` split at the gated product into `gate_w` and `out_w` (each with one 128 KiB weight resident), the gate
# backward with the forward's sigmoid gate saved (`gate_bwd_big_w`), and dy = [dgp | dv] [Wg; Wv] on cuBLAS ahead of the
# LayerNorm backward (`ln_bwd_w`); the pair side is `pair_wide.cu` at d_pair 512
BIG = ((256, 512, 32),)
NMAX = 1024  # the pair kernels hold one row of logits in shared memory
_EXT: dict[str, Any] = {}


def supported(d_msa: int, d_pair: int, n_head: int, d_hidden: int) -> bool:
    return n_head == H and ((d_msa in DMSA and d_pair in DPAIR and d_hidden in DHID) or (d_msa, d_pair, d_hidden) in BIG)


def is_big(d_msa: int, d_hidden: int) -> bool:
    """The value / gate widths that the BIG build serves (H * d_hidden > 128 or d_msa * H * d_hidden > 16384)."""
    return H * d_hidden > 128 or d_msa * H * d_hidden > 16384


def describe() -> str:
    return (f"d_msa in {DMSA}, d_pair in {DPAIR}, {H} heads x d_hidden in {DHID}, or (d_msa, d_pair, d_hidden) in {BIG} with "
            f"{H} heads")


def _load(name: str, src: str, flags: list[str]):
    if name in _EXT:
        return _EXT[name]
    from miniworld_engine.kernels._nvcc import load_extension as load

    csrc = Path(__file__).with_name("csrc")
    _EXT[name] = load(
        name=name,
        sources=[str(csrc / src)],
        extra_include_paths=[str(csrc)],
        extra_cuda_cflags=[
            "-O3",
            "-gencode=arch=compute_90a,code=sm_90a",
            "--use_fast_math",
            *flags,
        ],
        extra_cflags=["-O3"],
    )
    return _EXT[name]


# d_pair -> (warps per CTA, CTAs per SM) of the pair kernels
PAIR_CFG: dict[int, tuple[int, int]] = {128: (8, 2), 256: (4, 3), 384: (6, 1), 512: (4, 1)}   # 512: two 16 KiB tiles per warp


def _pair(d_pair: int):
    nwarp, minb = PAIR_CFG[d_pair]
    return _load(
        f"miniworld_pwa_pair_wide_{d_pair}_{nwarp}_{minb}",
        "pair_wide.cu",
        [f"-DPW_DZ={d_pair}", f"-DPW_NWARP={nwarp}", f"-DPW_MINB={minb}"],
    )


def _value(d_msa: int, d_hidden: int):
    return _load(
        f"miniworld_pwa_wide_{d_msa}_{d_hidden}",
        "pwa_wide.cu",
        [f"-DPW_D={d_msa}", f"-DPW_C={d_hidden}"],
    )


#: BIG build (d_msa 256 x 8 x 32) inference: ln_vg_w keeps y and gate_w reads it (True), or gate_w normalises m again (False)
BIG_INFER_Y = True


def _forward_math(
    args: list[torch.Tensor], eps_m: float, eps_z: float, p_drop: float, save: bool
):
    msa, pair, pm, lnm_w, lnm_b, wv, wg, lnz_w, wb, wo = args
    bf = torch.bfloat16
    m = msa[0]
    z = pair[0]
    d_msa, d_pair, hc = m.shape[2], z.shape[2], wv.shape[0]
    kv, kp = _value(d_msa, hc // H), _pair(d_pair)
    # the module has no pair-LayerNorm beta (a per-head constant on the logits, invisible to the softmax): the kernels take zeros
    w16 = kp.pair_fwd3(
        z,
        pm,
        lnz_w.detach().float().contiguous(),
        torch.zeros_like(lnz_w, dtype=torch.float32),
        eps_z,
        wb.detach(),
    )
    lnw, lnb = lnm_w.detach().float().contiguous(), lnm_b.detach().float().contiguous()
    big = is_big(d_msa, hc // H)
    v, y, stats = kv.ln_vg_w(
        m, lnw, lnb, wv.detach().to(bf).contiguous(), eps_m, save or (big and BIG_INFER_Y)
    )  # inference keeps no y: gate_out_w normalises m again (the BIG build: BIG_INFER_Y)
    o = torch.bmm(w16, v)  # [H, N, N] x [H, N, S*C]: the contraction on cuBLAS
    dmask, dscale = None, 1.0
    if (
        p_drop > 0
    ):  # the module's drop_msa: one keep-mask per (token, channel) shared over the MSA rows
        dmask = (torch.rand(m.shape[1], d_msa, device=m.device, dtype=bf) > p_drop).to(
            bf
        )
        dscale = 1.0 / (1.0 - p_drop)
    wg16, wo16 = wg.detach().to(bf).contiguous(), wo.detach().to(bf).contiguous()
    if is_big(d_msa, hc // H):  # gate_out_w split at the gated product; training keeps the sigmoid gate g for the backward
        if not save:
            a, _ = kv.gate_w(o, y, wg16, False) if BIG_INFER_Y else kv.gate_w(o, m, wg16, False, lnw, lnb, eps_m)
            return kv.out_w(a, m, wo16, dmask, dscale)
        a, g = kv.gate_w(o, y, wg16, True)
        return kv.out_w(a, m, wo16, dmask, dscale), w16, v, y, stats, o, dmask, g
    if not save:
        return kv.gate_out_w(o, m, m, wg16, wo16, dmask, dscale, lnw, lnb, eps_m)
    out = kv.gate_out_w(o, y, m, wg16, wo16, dmask, dscale)
    return out, w16, v, y, stats, o, dmask, None


def _forward_fake(args, eps_m, eps_z, p_drop):
    m = args[0]
    _, s, n, d = m.shape
    hc = args[5].shape[0]
    dev, bf = m.device, torch.bfloat16
    return [
        torch.empty((s, n, d), device=dev, dtype=bf),
        torch.empty((H, n, n), device=dev, dtype=bf),
        torch.empty((H, n, s * (hc // H)), device=dev, dtype=bf),
        torch.empty((s, n, d), device=dev, dtype=bf),
        torch.empty((s, n, 2), device=dev, dtype=torch.float32),
        torch.empty((H, n, s * (hc // H)), device=dev, dtype=bf),
        torch.empty((n, d) if p_drop else (0,), device=dev, dtype=bf),
        torch.empty((s, n, hc) if is_big(d, hc // H) else (0,), device=dev, dtype=bf),
    ]


@opaque(fake=_forward_fake, name="pwa_h100w_fwd")
def _forward_op(
    args: list[torch.Tensor], eps_m: float, eps_z: float, p_drop: float
) -> list[torch.Tensor]:
    out, w16, v, y, stats, o, dmask, g = _forward_math(args, eps_m, eps_z, p_drop, True)
    return [
        out,
        w16,
        v,
        y,
        stats,
        o,
        dmask if dmask is not None else args[0].new_empty((0,)),
        g if g is not None else args[0].new_empty((0,)),   # the BIG build's sigmoid gate [S, N, HC]
    ]


def _infer_fake(args, eps_m, eps_z):
    return torch.empty_like(args[0][0])


@opaque(fake=_infer_fake, name="pwa_h100w_infer")
def _inference_op(args: list[torch.Tensor], eps_m: float, eps_z: float) -> torch.Tensor:
    return _forward_math(args, eps_m, eps_z, 0.0, False)


def _backward_fake(args, kept, dres, eps_m, eps_z, p_drop):
    return [
        torch.empty_like(args[0][0]),
        torch.empty_like(args[1][0]),
        *[torch.empty_like(args[i]) for i in (3, 4, 5, 6, 7, 8, 9)],
    ]


@opaque(fake=_backward_fake, name="pwa_h100w_bwd")
def _backward_op(
    args: list[torch.Tensor],
    kept: list[torch.Tensor],
    dres: torch.Tensor,
    eps_m: float,
    eps_z: float,
    p_drop: float,
) -> list[torch.Tensor]:
    msa, pair, _pm, lnm_w, lnm_b, wv, wg, lnz_w, wb, wo = args
    w16, v, y, stats, o, dmask, g = kept
    bf = torch.bfloat16
    m, z = msa[0], pair[0]
    s, n, d_msa = m.shape
    d_pair, hc = z.shape[2], wv.shape[0]
    kv, kp = _value(d_msa, hc // H), _pair(d_pair)
    wg16, wv16, wo16 = (
        wg.to(bf).contiguous(),
        wv.to(bf).contiguous(),
        wo.to(bf).contiguous(),
    )
    dres = dres.contiguous()
    # the residual's gradient is dres itself; the update's is dres * mask / (1 - p) (what autograd of x * mask / (1-p) gives)
    dout = (
        dres
        if not p_drop
        else (dres * dmask[None] * (1.0 / (1.0 - p_drop))).contiguous()
    )
    dgv = torch.empty(
        (s, n, 2 * hc), device=m.device, dtype=bf
    )  # [dgp | dv], natural layout
    big = is_big(d_msa, hc // H)
    if big:  # the gate is the forward's g (no Wg on chip)
        do, go = kv.gate_bwd_big_w(o, g, dout, wo16.t().contiguous(), dgv)
    else:
        do, go = kv.gate_bwd_w(o, y, dout, wg16, wo16.t().contiguous(), dgv)
    dwo = torch.mm(dout.view(s * n, d_msa).t(), go.view(s * n, hc))  # [D, HC]
    dv = torch.bmm(w16.transpose(1, 2), do)  # head-major [H, N, S*C]
    dw = torch.bmm(do, v.transpose(1, 2))  # [H, N, N]; PyTorch's dw is bf16 too
    if big:
        # dv into the natural [dgp | dv] buffer, dy = [dgp | dv] [Wg; Wv] on cuBLAS in fp32 (a 256 KiB weight: no room on chip next to
        # the tiles), then the LayerNorm backward + the residual gradient
        c = hc // H
        dgv[..., hc:].view(s, n, H, c).copy_(dv.view(H, n, s, c).permute(2, 1, 0, 3))
        dy = torch.mm(dgv.view(s * n, 2 * hc), torch.cat([wg16, wv16], 0), out_dtype=torch.float32)
        dm, dlw, dlb = kv.ln_bwd_w(dy, m, stats, dres, lnm_w.detach().float().contiguous())
    else:
        dm, dlw, dlb = kv.dm_bwd_w(
            dv,
            dgv,
            m,
            stats,
            dres,
            torch.cat([wg16, wv16], 0).t().contiguous(),
            lnm_w.detach().float().contiguous(),
        )
    dwgv = torch.mm(
        dgv.view(s * n, 2 * hc).t(), y.view(s * n, d_msa)
    )  # [2 HC, D]: dWg over dWv
    dz, dwb, dzw, _ = kp.pair_bwd3(
        z,
        w16,
        dw,
        lnz_w.detach().float().contiguous(),
        torch.zeros_like(lnz_w, dtype=torch.float32),
        eps_z,
        wb.detach(),
    )
    cp = (
        lambda t, ref: t.to(ref.dtype, copy=True)
    )  # a custom op's returns may not alias each other (dlw / dlb, dWg / dWv are slices)
    return [
        dm,
        dz,
        cp(dlw, lnm_w),
        cp(dlb, lnm_b),
        cp(dwgv[hc:], wv),
        cp(dwgv[:hc], wg),
        cp(dzw, lnz_w),
        cp(dwb, wb),
        cp(dwo, wo),
    ]


class PwaWideFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, *args):
        tensors = list(args[:10])
        ctx.eps_m, ctx.eps_z, ctx.p_drop = args[10:]
        out, *kept = _forward_op(tensors, ctx.eps_m, ctx.eps_z, ctx.p_drop)
        ctx.save_for_backward(*tensors, *kept)
        return out[None]

    @staticmethod
    def backward(ctx, dy):
        vals = ctx.saved_tensors
        g = _backward_op(
            list(vals[:10]), list(vals[10:]), dy[0], ctx.eps_m, ctx.eps_z, ctx.p_drop
        )
        return (g[0][None], g[1][None], None, *g[2:], None, None, None)


def _prepare(module, msa: torch.Tensor, pair: torch.Tensor, mask: torch.Tensor | None):
    """Pad the length to a multiple of 16 (masked tokens); -> (msa, pair, key mask [N] bf16, original N)."""
    n = msa.shape[2]
    pn = -n % 16
    if pn:
        msa = F.pad(msa, (0, 0, 0, pn))
        pair = F.pad(pair, (0, 0, 0, pn, 0, pn))
    if mask is None:
        km = torch.ones(n + pn, dtype=torch.bfloat16, device=msa.device)
        if pn:
            km[n:] = 0
    else:
        km = F.pad(mask[0].to(torch.bfloat16), (0, pn))
    return msa.contiguous(), pair.contiguous(), km.contiguous(), n


def pair_weighted_averaging(
    module, msa: torch.Tensor, pair: torch.Tensor, mask: torch.Tensor | None
) -> torch.Tensor:
    """msa + MSAPairWeightedAveraging(msa, pair, mask) with grad: the residual and the row dropout are inside the kernel."""
    eps_m = float(getattr(module.ln_msa, "eps", 1e-5))
    eps_z = float(getattr(module.ln_pair, "eps", 1e-5))
    p_drop = float(module.drop_msa.p_drop) if module.training else 0.0
    m, z, km, n = _prepare(module, msa, pair, mask)
    out = PwaWideFn.apply(
        m,
        z,
        km,
        module.ln_msa.weight,
        module.ln_msa.bias,
        module.to_value.weight,
        module.to_gate.weight,
        module.ln_pair.weight,
        module.to_bias.weight,
        module.to_out.weight,
        eps_m,
        eps_z,
        p_drop,
    )
    return out[:, :, :n] if out.shape[2] != n else out


def pair_weighted_averaging_inference(
    module, msa: torch.Tensor, pair: torch.Tensor, mask: torch.Tensor | None
) -> torch.Tensor:
    eps_m = float(getattr(module.ln_msa, "eps", 1e-5))
    eps_z = float(getattr(module.ln_pair, "eps", 1e-5))
    m, z, km, n = _prepare(module, msa, pair, mask)
    args = [
        m,
        z,
        km,
        module.ln_msa.weight,
        module.ln_msa.bias,
        module.to_value.weight,
        module.to_gate.weight,
        module.ln_pair.weight,
        module.to_bias.weight,
        module.to_out.weight,
    ]
    out = _inference_op(args, eps_m, eps_z)[None]
    return out[:, :, :n] if out.shape[2] != n else out
