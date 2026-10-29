"""Chai-1 OuterProductMean (``groups = 8``, d_hidden 64, product LayerNorm): fused CUDA passes on H100 (``cuda/chai.cu``) and B200
(``integrations/csrc/sm100/opm_wide_sm100.cu``: tcgen05 prologue / epilogue / prologue backward, a CUDA-core product-norm backward).

    prologue   LayerNorm of the MSA rows + both projections + mask -> a, b [8, S N, 8] (group-major)         ``cuda.chai_proj``
    products   O[g] = a_g^T b_g over the MSA rows: one cuBLAS batched GEMM                                  [8, N K, N K]
    epilogue   per pair: LayerNorm of the 512 products, out-projection (+ bias), + residual, one TMA + wgmma pass   ``cuda.chai_epilogue``

The product LayerNorm's affine folds into the out-projection: W' = W_out . gamma (columns), b' = b_out + W_out beta.
Backward: dy = dz W' and dW' = dz^T y, dbias = sum dz (cuBLAS); the LayerNorm backward scatters dy into the grouped layout (``cuda.chai_bwd``);
dA = b dO^T, dB = a dO (two batched GEMMs); the projection / LayerNorm backward of the MSA rows in torch.  dgamma = sum_o W dW', dbeta = W^T dbias.

B200 (sm_100a), the same stages: ``opm_chai_prologue`` (LayerNorm + both projections on the tensor core, D[s][channel], stored group-major),
the batched GEMM, ``opm_chai_epilogue`` (the pair tile of O gathered by 5-D TMA boxes into the K-major layout, normalised in shared
memory, one tcgen05 GEMM against W', y and the statistics saved for training), and in the backward ``opm_chai_lnbwd`` (the product
norm's backward into the grouped layout) and ``opm_chai_prologue_bwd`` (dm and the projection / LayerNorm gradients, R and the masked
column sums on the tensor core).  A length that is not a multiple of 32 / a depth that is not a multiple of 128 is padded (masked rows add
no product; padded pairs are cut away).  B200 also serves the interchain mask (chain ids into the epilogue: a pair of two chains gets no
update; in training its y and rstd are saved as zero and its dbias weight is zero) and a batch of MSA stacks (one stack at a time).
"""

from __future__ import annotations

import torch

from miniworld_engine.kernels._compile import opaque

G, K, P, CM = 8, 8, 512, 64


def _sm100(t: torch.Tensor) -> bool:
    return t.is_cuda and torch.cuda.get_device_capability(t.device) == (10, 0)


def serves_dims(msa: torch.Tensor, groups: int, group_width: int, d_pair: int, *, mask: torch.Tensor | None = None, projection_bias: bool = False) -> bool:
    if not msa.is_cuda or msa.dtype != torch.bfloat16 or msa.ndim != 4 or msa.shape[0] < 1:
        return False
    if _sm100(msa):   # B200: any length / depth (padded), c_z 128 / 256, a batch (one stack at a time)
        _, s, n, cm = msa.shape
        return (groups == G and group_width == K and cm == CM and n > 0 and s > 0 and d_pair in (128, 256) and not projection_bias
                and (mask is None or mask.shape == msa.shape[:3]))
    if torch.cuda.get_device_capability(msa.device) != (9, 0) or msa.shape[0] != 1:
        return False
    _, s, n, cm = msa.shape
    return (groups == G and group_width == K and cm == CM and n % 64 == 0 and s > 0 and d_pair % 128 == 0 and not projection_bias
            and (mask is None or mask.shape == msa.shape[:3]))


def serves(module, msa: torch.Tensor, mask: torch.Tensor | None, *, training: bool = False, interchain: bool = False) -> bool:
    if training and interchain and not _sm100(msa):   # B200: the epilogue applies it (and zeroes what the backward reads)
        return False
    return module.to_out.weight.shape[1] == P and serves_dims(msa, module.groups, module.group_width, module.to_out.weight.shape[0], mask=mask,
                                                            projection_bias=module.to_left.bias is not None)


def _fold(wo, gam, bet, bo):
    """(W' [CZ, 512] bf16, b' [CZ] fp32): the product LayerNorm's gamma / beta folded into the out-projection."""
    w = wo.float()
    return (w * gam.float()[None, :]).to(torch.bfloat16).contiguous(), (bo.float() + w @ bet.float()).contiguous()


def _forms(lnw, lnb, wl, wr, gam, bet, wo, bo):
    """The kernel-dtype forms of the weights, made once while the weights are unchanged (`_capture.param_forms`): (LayerNorm gamma / beta fp32,
    [W_left; W_right] bf16, W' bf16, b' fp32). With fp32 master parameters this was a handful of cast kernels per call."""
    from miniworld_engine.kernels import _capture

    def build():
        wf, bf = _fold(wo, gam, bet, bo)
        return lnw.float().contiguous(), lnb.float().contiguous(), torch.cat([wl, wr]).to(torch.bfloat16).contiguous(), wf, bf

    return _capture.param_forms("chai_forms", [lnw, lnb, wl, wr, gam, bet, wo, bo], build)


def _front(msa, mask, forms, eps, save):
    from miniworld_engine.kernels.outer_product_mean import cuda

    _, s, n, _ = msa.shape
    x = msa[0].reshape(s * n, CM).contiguous()
    mk = None
    if mask.numel():
        mk = mask[0].reshape(s * n).contiguous()
        if mk.dtype != torch.bool:
            mk = mk != 0
    lnw32, lnb32, w, _, _ = forms
    a, b, st1 = cuda.chai_proj(x, mk, lnw32, lnb32, w, eps, save)
    o = torch.bmm(a.view(G, s, n * K).transpose(1, 2), b.view(G, s, n * K))     # [8, N K, N K]
    return a, b, st1, o


def _fwd_op_fake(args, eps, eps_prod, train):
    """Shapes of ``_fwd_op``: z [1, N, N, CZ] bf16; with ``train`` also a, b [8, S N, 8], y [N N, 512], the epilogue and prologue statistics."""
    msa, wo = args[0], args[8]
    _, s, n, _ = msa.shape
    bf = torch.bfloat16
    new = lambda *shape, dt=bf: msa.new_empty(shape, dtype=dt)
    ws = new(n * n) if train and args[11].numel() else new(0)
    if not train:
        return [new(1, n, n, wo.shape[0]), new(0), new(0), new(0), new(0, dt=torch.float32), new(0, dt=torch.float32), ws]
    return [new(1, n, n, wo.shape[0]), new(G, s * n, K), new(G, s * n, K), new(n * n, P), new(n * n, 2, dt=torch.float32), new(s * n, 2, dt=torch.float32),
            ws]


def _ext100():
    from miniworld_engine.integrations import opm_train

    return opm_train._ext_wide()


def _mask100(msa, mask):
    """The bool [S, N] mask of the B200 kernels (all rows valid without one)."""
    if not mask.numel():
        return torch.ones(msa.shape[1:3], dtype=torch.bool, device=msa.device)
    mk = mask[0].contiguous()
    return mk if mk.dtype == torch.bool else mk != 0


def _fwd_sm100(args, eps, eps_prod, train):
    msa, mask, lnw, lnb, wl, wr, gam, bet, wo, bo, residual, asym = args
    _, s, n, _ = msa.shape
    ext = _ext100()
    lnw32, lnb32, wlr, wf, bf = _forms(lnw, lnb, wl, wr, gam, bet, wo, bo)
    a, b, st1 = ext.opm_chai_prologue(msa[0].contiguous(), _mask100(msa, mask), lnw32, lnb32, float(eps), wlr[:G * K], wlr[G * K:], bool(train))
    o = torch.bmm(a.view(G, s, n * K).transpose(1, 2), b.view(G, s, n * K))     # [8, N K, N K]
    z, y, st, ws = ext.opm_chai_epilogue(o, wf, bf, residual if residual.numel() else None, bool(train), float(eps_prod),
                                         asym if asym.numel() else None)
    if not train:
        return [z, msa.new_empty(0), msa.new_empty(0), msa.new_empty(0), msa.new_empty(0, dtype=torch.float32), msa.new_empty(0, dtype=torch.float32), ws]
    return [z, a, b, y, st, st1.view(-1, 2), ws]    # st1: [N, S, 2] (token-major), viewed as the [S N, 2] the fake declares


@opaque(fake=_fwd_op_fake, name="outer_product_mean_chai_fwd")
def _fwd_op(args: list[torch.Tensor], eps: float, eps_prod: float, train: bool) -> list[torch.Tensor]:
    """The Chai-1 OPM forward: prologue, grouped products, fused epilogue. args = [msa, mask, lnw, lnb, wl, wr, gamma, beta, wo, bo, residual,
    asym] (empty tensors for no mask / residual / chain ids; chain ids only on B200)."""
    from miniworld_engine.kernels.outer_product_mean import cuda
    if _sm100(args[0]):
        return _fwd_sm100(args, eps, eps_prod, train)
    msa, mask, lnw, lnb, wl, wr, gam, bet, wo, bo, residual, _ = args
    forms = _forms(lnw, lnb, wl, wr, gam, bet, wo, bo)
    a, b, st1, o = _front(msa, mask, forms, eps, train)
    wf, bf = forms[3], forms[4]
    z, y, st = cuda.chai_epilogue(o, wf, bf, residual if residual.numel() else None, train, eps_prod)
    if not train:
        return [z, msa.new_empty(0), msa.new_empty(0), msa.new_empty(0), msa.new_empty(0, dtype=torch.float32), msa.new_empty(0, dtype=torch.float32),
                msa.new_empty(0)]
    return [z, a, b, y, st, st1, msa.new_empty(0)]


def _bwd_op_fake(args, kept, dz, eps, eps_prod):
    """Shapes of ``_bwd_op``: one gradient per differentiable argument (msa, LayerNorm, projections, product norm, out-projection), in the argument's dtype."""
    return [torch.empty_like(args[i], dtype=args[i].dtype) for i in (0, 2, 3, 4, 5, 6, 7, 8, 9)]


@opaque(fake=_bwd_op_fake, name="outer_product_mean_chai_bwd")
def _bwd_op(args: list[torch.Tensor], kept: list[torch.Tensor], dz: torch.Tensor, eps: float, eps_prod: float) -> list[torch.Tensor]:
    """Gradients of ``_fwd_op`` (train): [dmsa, dlnw, dlnb, dwl, dwr, dgamma, dbeta, dwo, dbo] from dz and the saved a, b, y, statistics."""
    from miniworld_engine.kernels.outer_product_mean import cuda

    msa, mask, lnw, lnb, wl, wr, gam, bet, wo, bo, _, _ = args
    a, b, y, st, st1, ws = kept
    bf, f32 = torch.bfloat16, torch.float32
    _, s, n, _ = msa.shape
    t = s * n
    cz = wo.shape[0]
    dz2 = dz.reshape(n * n, cz).contiguous()
    lnw32, lnb32, wlr, wf, _ = _forms(lnw, lnb, wl, wr, gam, bet, wo, bo)
    dy = torch.mm(dz2, wf)                                                          # [N N, 512] = dz W'
    # sum over the pairs (the interchain mask: over the same-chain pairs, the epilogue's 1 / 0 weights; y and rstd are zero elsewhere)
    dbo = torch.mm(ws.view(1, n * n) if ws.numel() else dz2.new_ones((1, n * n)), dz2, out_dtype=f32).reshape(-1)
    dwp = torch.mm(dz2.t(), y, out_dtype=f32)                                       # dW' = dz^T y
    w32 = wo.float()
    dwo = dwp * gam.float()[None, :] + dbo[:, None] * bet.float()[None, :]                  # W' = W gamma, b' = b + W beta
    dgam = (w32 * dwp).sum(0)
    dbet = w32.t() @ dbo
    if _sm100(dz):
        ext = _ext100()
        do = ext.opm_chai_lnbwd(dy, y, st, n)
        del dy
        da = torch.bmm(do, b.view(G, s, n * K).transpose(1, 2))                     # dA^T [8, N K, S]: the kernel's MSA-row-contiguous tiles
        db = torch.bmm(do.transpose(1, 2), a.view(G, s, n * K).transpose(1, 2))     # dB^T
        del do
        dx, dwl, dwr, dlw, dlb = ext.opm_chai_prologue_bwd(da, db, msa[0].contiguous(), st1.view(n, s, 2), _mask100(msa, mask), lnw32, lnb32,
                                                           wlr[:G * K], wlr[G * K:])
        return [dx, dlw.to(lnw.dtype), dlb.to(lnb.dtype), dwl.to(wl.dtype), dwr.to(wr.dtype), dgam.to(gam.dtype), dbet.to(bet.dtype),
                dwo.to(wo.dtype), dbo.to(bo.dtype)]
    do = cuda.chai_bwd(dy, y, st, n)
    del dy
    da = torch.bmm(b.view(G, s, n * K), do.transpose(1, 2))                         # [8, S, N K]
    db = torch.bmm(a.view(G, s, n * K), do)
    del do
    x = msa[0].reshape(t, CM).contiguous()
    mk = None
    if mask.numel():
        mk = mask[0].reshape(t).contiguous()
        if mk.dtype != torch.bool:
            mk = mk != 0
    dcat, ln16, dx, part = cuda.chai_projbwd(x, mk, da.view(G, t, K), db.view(G, t, K), lnw32, lnb32, st1, wlr)
    dw = torch.mm(dcat.t(), ln16, out_dtype=f32)                                    # [128, 64]
    dlw, dlb = part[:, :CM].sum(0), part[:, CM:].sum(0)
    return [dx.view(1, s, n, CM).clone(), dlw.to(lnw.dtype), dlb.to(lnb.dtype), dw[:G * K].to(wl.dtype).clone(), dw[G * K:].to(wr.dtype).clone(),
            dgam.to(gam.dtype), dbet.to(bet.dtype), dwo.to(wo.dtype), dbo.to(bo.dtype)]


class _ChaiFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, eps, eps_prod, *args):
        tensors = list(args)
        z, a, b, y, st, st1, ws = _fwd_op(tensors, eps, eps_prod, True)
        ctx.eps, ctx.eps_prod, ctx.has_residual = eps, eps_prod, bool(tensors[10].numel())
        ctx.save_for_backward(*tensors, a, b, y, st, st1, ws)
        return z

    @staticmethod
    def backward(ctx, dz):
        vals = ctx.saved_tensors
        grads = _bwd_op(list(vals[:12]), list(vals[12:]), dz.contiguous(), ctx.eps, ctx.eps_prod)
        # args: msa, mask, lnw, lnb, wl, wr, gam, bet, wo, bo, residual, asym
        return (None, None, grads[0], None, *grads[1:], dz if ctx.has_residual else None, None)


def forward(module, msa, mask, token_asym_id=None, residual=None):
    interchain = module.mask_interchain and token_asym_id is not None
    if msa.shape[0] > 1:   # B200: one MSA stack at a time (each its own autograd node)
        return torch.cat([forward(module, msa[b:b + 1], None if mask is None else mask[b:b + 1],
                                  token_asym_id[b:b + 1] if interchain else None, None if residual is None else residual[b:b + 1])
                          for b in range(msa.shape[0])])
    n = msa.shape[2]
    empty = msa.new_empty(0)
    pn = ps = 0
    if _sm100(msa):   # B200: a length to a multiple of 32, a depth to a multiple of 128, with masked rows (no product)
        pn, ps = -n % 32, -msa.shape[1] % 128
        if pn or ps:
            if mask is None:
                mask = torch.ones(msa.shape[:3], dtype=torch.bool, device=msa.device)
            msa = torch.nn.functional.pad(msa, (0, 0, 0, pn, 0, ps))
            mask = torch.nn.functional.pad(mask, (0, pn, 0, ps))
    # B200: the chain ids go into the epilogue (padded tokens -1: their pairs are cut away), which also takes the residual
    asym = empty.to(torch.int32)
    if interchain and _sm100(msa):
        asym = torch.nn.functional.pad(token_asym_id[0].to(torch.int32), (0, pn), value=-1).contiguous()
    fuse = residual is not None and (not interchain or asym.numel() > 0) and residual.dtype == msa.dtype and not pn
    args = [msa.contiguous(), empty if mask is None else mask, module.ln_msa.weight, module.ln_msa.bias, module.to_left.weight,
            module.to_right.weight, module.product_norm.weight, module.product_norm.bias, module.to_out.weight, module.to_out.bias,
            residual.contiguous() if fuse else empty, asym]
    eps, eps_prod = float(module.ln_msa.eps), float(module.product_norm.eps)
    if torch.is_grad_enabled():
        z = _ChaiFn.apply(eps, eps_prod, *args)
    else:
        z = _fwd_op(args, eps, eps_prod, False)[0]
    z = z.view(1, n + pn, n + pn, -1)
    if pn:
        z = z[:, :n, :n]
    if interchain and not asym.numel():
        z = z * (token_asym_id[:, :, None] == token_asym_id[:, None, :])[..., None].to(z.dtype)
    return residual + z if residual is not None and not fuse else z
