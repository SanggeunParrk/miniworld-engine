"""Chai-1 OuterProductMean (``groups = 8``, d_hidden 64, product LayerNorm) on the B200 (sm_100a): the ``cuda`` backend's route at
capability (10, 0). The stages of ``chai.py`` on ``integrations/csrc/sm100/opm_wide_sm100.cu``:

    prologue   ``opm_chai_prologue``: LayerNorm + both projections on the tensor core (D[s][channel]) x mask, stored group-major a, b
    products   O[g] = a_g^T b_g over the MSA rows: one cuBLAS batched GEMM                                  [8, N K, N K]
    epilogue   ``opm_chai_epilogue``: the pair tile of O gathered by TMA boxes into the K-major layout, the product LayerNorm in shared
               memory, one tcgen05 GEMM against W' (= W_out gamma), + b' (+ residual); y and the statistics saved for training
    backward   dy = dz W', dW' = dz^T y, dbias (cuBLAS); ``opm_chai_lnbwd`` (the product norm's backward into the grouped layout);
               dA, dB (two batched GEMMs); ``opm_chai_prologue_bwd`` (dm and the projection / LayerNorm gradients)

A length that is not a multiple of 32 / a depth that is not a multiple of 128 is padded with masked rows (no product; padded pairs are cut
away). The interchain mask (chain ids into the epilogue: a pair of two chains gets no update, and in training its y and rstd are saved as
zero and its dbias weight is zero) and a batch of MSA stacks (one stack at a time) are served. From the B200 branch
(``origin/b200/ladder-final``, ``docs/gpus/b200/opm/opm.md``).
"""

from __future__ import annotations

import torch

from miniworld_engine.kernels import _cast_cuda
from miniworld_engine.kernels._compile import opaque
from miniworld_engine.kernels.outer_product_mean.chai import CM, G, K, P, _forms

D_PAIR = (128, 256)


def serves(module, msa: torch.Tensor, mask: torch.Tensor | None) -> bool:
    if not msa.is_cuda or msa.dtype != torch.bfloat16 or msa.ndim != 4 or torch.cuda.get_device_capability(msa.device) != (10, 0):
        return False
    _, s, n, cm = msa.shape
    return (
        module.groups == G
        and module.group_width == K
        and module.to_out.weight.shape[1] == P
        and cm == CM
        and n > 0
        and s > 0
        and module.to_out.weight.shape[0] in D_PAIR
        and module.to_left.bias is None
        and (mask is None or mask.shape == msa.shape[:3])
    )


def _ext():
    from miniworld_engine.kernels.integrations import opm_sm100

    return opm_sm100._ext_wide()


def _mask(msa: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """The bool [S, N] mask of the kernels (all rows valid without one)."""
    if not mask.numel():
        return torch.ones(msa.shape[1:3], dtype=torch.bool, device=msa.device)
    mk = mask[0].contiguous()
    return mk if mk.dtype == torch.bool else mk != 0


def _fwd_fake(args, eps, eps_prod, train):
    msa, wo = args[0], args[8]
    _, s, n, _ = msa.shape
    bf = torch.bfloat16
    new = lambda *shape, dt=bf: msa.new_empty(shape, dtype=dt)
    ws = new(n * n) if train and args[11].numel() else new(0)
    if not train:
        return [new(1, n, n, wo.shape[0]), new(0), new(0), new(0), new(0, dt=torch.float32), new(0, dt=torch.float32), ws]
    return [
        new(1, n, n, wo.shape[0]),
        new(G, s * n, K),
        new(G, s * n, K),
        new(n * n, P),
        new(n * n, 2, dt=torch.float32),
        new(s * n, 2, dt=torch.float32),
        ws,
    ]


@opaque(fake=_fwd_fake, name="outer_product_mean_chai_sm100_fwd")
def _fwd_op(args: list[torch.Tensor], eps: float, eps_prod: float, train: bool) -> list[torch.Tensor]:
    """args = [msa, mask, lnw, lnb, wl, wr, gamma, beta, wo, bo, residual, asym] (empty tensors: no mask / residual / chain ids)
    -> [z, a, b, y, stats, stats1, ws] (training keeps a, b, y and the statistics; ws: the dbias weights of the interchain mask)."""
    msa, mask, lnw, lnb, wl, wr, gam, bet, wo, bo, residual, asym = args
    _, s, n, _ = msa.shape
    ext = _ext()
    lnw32, lnb32, wlr, wf, bf = _forms(lnw, lnb, wl, wr, gam, bet, wo, bo, infer=not train)
    a, b, st1 = ext.opm_chai_prologue(
        msa[0].contiguous(), _mask(msa, mask), lnw32, lnb32, float(eps), wlr[: G * K], wlr[G * K :], bool(train)
    )
    o = torch.bmm(a.view(G, s, n * K).transpose(1, 2), b.view(G, s, n * K))  # [8, N K, N K]
    z, y, st, ws = ext.opm_chai_epilogue(
        o, wf, bf, residual if residual.numel() else None, bool(train), float(eps_prod), asym if asym.numel() else None
    )
    z = z.view(1, n, n, wo.shape[0])
    if not train:
        e = msa.new_empty(0)
        return [z, e, e.clone(), e.clone(), e.new_empty(0, dtype=torch.float32), e.new_empty(0, dtype=torch.float32), msa.new_empty(0)]
    return [z, a, b, y, st, st1.view(-1, 2), ws]  # st1: [N, S, 2] (token-major), viewed as [S N, 2]


def _bwd_fake(args, kept, dz, eps, eps_prod):
    return [torch.empty_like(args[i]) for i in (0, 2, 3, 4, 5, 6, 7, 8, 9)]


@opaque(fake=_bwd_fake, name="outer_product_mean_chai_sm100_bwd")
def _bwd_op(args: list[torch.Tensor], kept: list[torch.Tensor], dz: torch.Tensor, eps: float, eps_prod: float) -> list[torch.Tensor]:
    """[dmsa, dlnw, dlnb, dwl, dwr, dgamma, dbeta, dwo, dbo] from dz and the saved a, b, y, statistics."""
    msa, mask, lnw, lnb, wl, wr, gam, bet, wo, bo, _, _ = args
    a, b, y, st, st1, ws = kept
    f32 = torch.float32
    _, s, n, _ = msa.shape
    cz = wo.shape[0]
    ext = _ext()
    dz2 = dz.reshape(n * n, cz).contiguous()
    lnw32, lnb32, wlr, wf, _ = _forms(lnw, lnb, wl, wr, gam, bet, wo, bo)
    dy = torch.mm(dz2, wf)  # [N N, 512] = dz W'
    # sum over the pairs (the interchain mask: over the same-chain pairs, the epilogue's 1 / 0 weights; y and rstd are zero elsewhere)
    dbo = torch.mm(ws.view(1, n * n) if ws.numel() else dz2.new_ones((1, n * n)), dz2, out_dtype=f32).reshape(-1)
    dwp = torch.mm(dz2.t(), y, out_dtype=f32)  # dW' = dz^T y
    w32 = wo.float()
    dwo = dwp * gam.float()[None, :] + dbo[:, None] * bet.float()[None, :]  # W' = W gamma, b' = b + W beta
    dgam = (w32 * dwp).sum(0)
    dbet = w32.t() @ dbo
    do = ext.opm_chai_lnbwd(dy, y, st, n)
    del dy
    da = torch.bmm(do, b.view(G, s, n * K).transpose(1, 2))  # dA^T [8, N K, S]: the kernel's MSA-row-contiguous tiles
    db = torch.bmm(do.transpose(1, 2), a.view(G, s, n * K).transpose(1, 2))  # dB^T
    del do
    dx, dwl, dwr, dlw, dlb = ext.opm_chai_prologue_bwd(
        da, db, msa[0].contiguous(), st1.view(n, s, 2), _mask(msa, mask), lnw32, lnb32, wlr[: G * K], wlr[G * K :]
    )
    return [
        dx.view(1, s, n, CM),
        # the parameter gradients in their parameters' dtypes, new tensors (a custom op's returns may not alias each other), ONE launch
        *_cast_cuda.cast([dlw, dlb, dwl, dwr, dgam, dbet, dwo, dbo], [p.dtype for p in (lnw, lnb, wl, wr, gam, bet, wo, bo)], force=True),
    ]


class _ChaiSm100Fn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, eps, eps_prod, *args):
        tensors = list(args)
        z, *kept = _fwd_op(tensors, eps, eps_prod, True)
        ctx.eps, ctx.eps_prod, ctx.has_residual = eps, eps_prod, bool(tensors[10].numel())
        ctx.save_for_backward(*tensors, *kept)
        return z

    @staticmethod
    def backward(ctx, dz):
        vals = ctx.saved_tensors
        g = _bwd_op(list(vals[:12]), list(vals[12:]), dz.to(torch.bfloat16).contiguous(), ctx.eps, ctx.eps_prod)
        # args: msa, mask, lnw, lnb, wl, wr, gam, bet, wo, bo, residual, asym
        return (None, None, g[0], None, *g[1:], dz if ctx.has_residual else None, None)


def forward(module, msa, mask, token_asym_id=None, residual=None):
    interchain = module.mask_interchain and token_asym_id is not None
    if msa.shape[0] > 1:  # one MSA stack at a time (each its own autograd node)
        return torch.cat(
            [
                forward(
                    module,
                    msa[b : b + 1],
                    None if mask is None else mask[b : b + 1],
                    token_asym_id[b : b + 1] if interchain and token_asym_id is not None else None,
                    None if residual is None else residual[b : b + 1],
                )
                for b in range(msa.shape[0])
            ]
        )
    n = msa.shape[2]
    pn, ps = -n % 32, -msa.shape[1] % 128
    if pn or ps:  # masked rows (no product)
        if mask is None:
            mask = torch.ones(msa.shape[:3], dtype=torch.bool, device=msa.device)
        msa = torch.nn.functional.pad(msa, (0, 0, 0, pn, 0, ps))
        mask = torch.nn.functional.pad(mask, (0, pn, 0, ps))
    empty = msa.new_empty(0)
    # the chain ids go into the epilogue (padded tokens -1: their pairs are cut away)
    asym = empty.to(torch.int32)
    if interchain and token_asym_id is not None:
        asym = torch.nn.functional.pad(token_asym_id[0].to(torch.int32), (0, pn), value=-1).contiguous()
    fuse = residual is not None and residual.dtype == msa.dtype and not pn
    args = [
        msa.contiguous(),
        empty if mask is None else mask.contiguous(),
        module.ln_msa.weight,
        module.ln_msa.bias,
        module.to_left.weight,
        module.to_right.weight,
        module.product_norm.weight,
        module.product_norm.bias,
        module.to_out.weight,
        module.to_out.bias,
        residual.contiguous() if fuse and residual is not None else empty,
        asym,
    ]
    eps, eps_prod = float(module.ln_msa.eps), float(module.product_norm.eps)
    z = _ChaiSm100Fn.apply(eps, eps_prod, *args) if torch.is_grad_enabled() else _fwd_op(args, eps, eps_prod, False)[0]
    if pn:
        z = z[:, :n, :n]
    return residual + z if residual is not None and not fuse else z
