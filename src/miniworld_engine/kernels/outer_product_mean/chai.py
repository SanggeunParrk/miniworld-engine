"""Chai-1 OuterProductMean (``groups = 8``, d_hidden 64, product LayerNorm) on H100: fused CUDA passes (``cuda/chai.cu``).

    prologue   LayerNorm of the MSA rows + both projections + mask -> a, b [8, S N, 8] (group-major)         ``cuda.chai_proj``
    products   O[g] = a_g^T b_g over the MSA rows: one cuBLAS batched GEMM                                  [8, N K, N K]
    epilogue   per pair: LayerNorm of the 512 products, out-projection (+ bias), + residual, one TMA + wgmma pass   ``cuda.chai_epilogue``

The product LayerNorm's affine folds into the out-projection: W' = W_out . gamma (columns), b' = b_out + W_out beta.
Backward: dy = dz W' and dW' = dz^T y, dbias = sum dz (cuBLAS); the LayerNorm backward scatters dy into the grouped layout (``cuda.chai_bwd``);
dA = b dO^T, dB = a dO (two batched GEMMs); the projection / LayerNorm backward of the MSA rows in torch.  dgamma = sum_o W dW', dbeta = W^T dbias.
"""

from __future__ import annotations

import torch

from miniworld_engine.kernels._compile import opaque

G, K, P, CM = 8, 8, 512, 64


def serves_dims(msa: torch.Tensor, groups: int, group_width: int, d_pair: int, *, mask: torch.Tensor | None = None, projection_bias: bool = False) -> bool:
    if not msa.is_cuda or msa.dtype != torch.bfloat16 or msa.ndim != 4 or msa.shape[0] != 1:
        return False
    if torch.cuda.get_device_capability(msa.device) != (9, 0):
        return False
    _, s, n, cm = msa.shape
    return (groups == G and group_width == K and cm == CM and n % 64 == 0 and s > 0 and d_pair % 128 == 0 and not projection_bias
            and (mask is None or mask.shape == msa.shape[:3]))


def serves(module, msa: torch.Tensor, mask: torch.Tensor | None, *, training: bool = False, interchain: bool = False) -> bool:
    if training and interchain:
        return False
    return module.to_out.weight.shape[1] == P and serves_dims(msa, module.groups, module.group_width, module.to_out.weight.shape[0], mask=mask,
                                                            projection_bias=module.to_left.bias is not None)


def _fold(wo, gam, bet, bo):
    """(W' [CZ, 512] bf16, b' [CZ] fp32): the product LayerNorm's gamma / beta folded into the out-projection."""
    w = wo.float()
    return (w * gam.float()[None, :]).to(torch.bfloat16).contiguous(), (bo.float() + w @ bet.float()).contiguous()


def _front(msa, mask, lnw, lnb, wl, wr, eps, save):
    from miniworld_engine.kernels.outer_product_mean import cuda

    _, s, n, _ = msa.shape
    x = msa[0].reshape(s * n, CM).contiguous()
    mk = None
    if mask.numel():
        mk = mask[0].reshape(s * n).contiguous()
        if mk.dtype != torch.bool:
            mk = mk != 0
    w = torch.cat([wl, wr]).to(torch.bfloat16).contiguous()
    a, b, st1 = cuda.chai_proj(x, mk, lnw.float().contiguous(), lnb.float().contiguous(), w, eps, save)
    o = torch.bmm(a.view(G, s, n * K).transpose(1, 2), b.view(G, s, n * K))     # [8, N K, N K]
    return a, b, st1, o


def _fwd_fake(args, eps, eps_prod, train):
    msa, wo = args[0], args[8]
    _, s, n, _ = msa.shape
    bf = torch.bfloat16
    new = lambda *shape, dt=bf: msa.new_empty(shape, dtype=dt)   # noqa: E731
    if not train:
        return [new(1, n, n, wo.shape[0]), new(0), new(0), new(0), new(0, dt=torch.float32), new(0, dt=torch.float32)]
    return [new(1, n, n, wo.shape[0]), new(G, s * n, K), new(G, s * n, K), new(n * n, P), new(n * n, 2, dt=torch.float32), new(s * n, 2, dt=torch.float32)]


@opaque(fake=_fwd_fake, name="opm_chai_fwd")
def _fwd_op(args: list[torch.Tensor], eps: float, eps_prod: float, train: bool) -> list[torch.Tensor]:
    from miniworld_engine.kernels.outer_product_mean import cuda

    msa, mask, lnw, lnb, wl, wr, gam, bet, wo, bo, residual = args
    a, b, st1, o = _front(msa, mask, lnw, lnb, wl, wr, eps, train)
    wf, bf = _fold(wo, gam, bet, bo)
    z, y, st = cuda.chai_epilogue(o, wf, bf, residual if residual.numel() else None, train, eps_prod)
    if not train:
        return [z, msa.new_empty(0), msa.new_empty(0), msa.new_empty(0), msa.new_empty(0, dtype=torch.float32), msa.new_empty(0, dtype=torch.float32)]
    return [z, a, b, y, st, st1]


def _bwd_fake(args, kept, dz, eps, eps_prod):
    return [torch.empty_like(args[i], dtype=args[i].dtype) for i in (0, 2, 3, 4, 5, 6, 7, 8, 9)]


@opaque(fake=_bwd_fake, name="opm_chai_bwd")
def _bwd_op(args: list[torch.Tensor], kept: list[torch.Tensor], dz: torch.Tensor, eps: float, eps_prod: float) -> list[torch.Tensor]:
    from miniworld_engine.kernels.outer_product_mean import cuda

    msa, mask, lnw, lnb, wl, wr, gam, bet, wo, bo, residual = args
    a, b, y, st, st1 = kept
    bf, f32 = torch.bfloat16, torch.float32
    _, s, n, _ = msa.shape
    t = s * n
    cz = wo.shape[0]
    dz2 = dz.reshape(n * n, cz).contiguous()
    wf, _ = _fold(wo, gam, bet, bo)
    dy = torch.mm(dz2, wf)                                                          # [N N, 512] = dz W'
    dbo = torch.mm(dz2.new_ones((1, n * n)), dz2, out_dtype=f32).reshape(-1)        # sum over the pairs
    dwp = torch.mm(dz2.t(), y, out_dtype=f32)                                       # dW' = dz^T y
    w32 = wo.float()
    dwo = dwp * gam.float()[None, :] + dbo[:, None] * bet.float()[None, :]                  # W' = W gamma, b' = b + W beta
    dgam = (w32 * dwp).sum(0)
    dbet = w32.t() @ dbo
    do = cuda.chai_bwd(dy, y, st, n)
    del dy
    da = torch.bmm(b.view(G, s, n * K), do.transpose(1, 2))                         # [8, S, N K]
    db = torch.bmm(a.view(G, s, n * K), do)
    del do
    wlr = torch.cat([wl, wr]).to(bf).contiguous()
    x = msa[0].reshape(t, CM).contiguous()
    mk = None
    if mask.numel():
        mk = mask[0].reshape(t).contiguous()
        if mk.dtype != torch.bool:
            mk = mk != 0
    dcat, ln16, dx, part = cuda.chai_projbwd(x, mk, da.view(G, t, K), db.view(G, t, K), lnw.float().contiguous(), lnb.float().contiguous(), st1, wlr)
    dw = torch.mm(dcat.t(), ln16, out_dtype=f32)                                    # [128, 64]
    dlw, dlb = part[:, :CM].sum(0), part[:, CM:].sum(0)
    return [dx.view(1, s, n, CM).clone(), dlw.to(lnw.dtype), dlb.to(lnb.dtype), dw[:G * K].to(wl.dtype).clone(), dw[G * K:].to(wr.dtype).clone(),
            dgam.to(gam.dtype), dbet.to(bet.dtype), dwo.to(wo.dtype), dbo.to(bo.dtype)]


class _ChaiFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, eps, eps_prod, *args):
        tensors = list(args)
        z, a, b, y, st, st1 = _fwd_op(tensors, eps, eps_prod, True)
        ctx.eps, ctx.eps_prod, ctx.has_residual = eps, eps_prod, bool(tensors[10].numel())
        ctx.save_for_backward(*tensors, a, b, y, st, st1)
        return z

    @staticmethod
    def backward(ctx, dz):
        vals = ctx.saved_tensors
        grads = _bwd_op(list(vals[:11]), list(vals[11:]), dz.contiguous(), ctx.eps, ctx.eps_prod)
        # args: msa, mask, lnw, lnb, wl, wr, gam, bet, wo, bo, residual
        return (None, None, grads[0], None, *grads[1:], dz if ctx.has_residual else None)


def forward(module, msa, mask, token_asym_id=None, residual=None):
    n = msa.shape[2]
    interchain = module.mask_interchain and token_asym_id is not None
    empty = msa.new_empty(0)
    fuse = residual is not None and not interchain and residual.dtype == msa.dtype
    args = [msa.contiguous(), empty if mask is None else mask, module.ln_msa.weight, module.ln_msa.bias, module.to_left.weight,
            module.to_right.weight, module.product_norm.weight, module.product_norm.bias, module.to_out.weight, module.to_out.bias,
            residual.contiguous() if fuse else empty]
    eps, eps_prod = float(module.ln_msa.eps), float(module.product_norm.eps)
    if torch.is_grad_enabled():
        z = _ChaiFn.apply(eps, eps_prod, *args)
    else:
        z = _fwd_op(args, eps, eps_prod, False)[0]
    z = z.view(1, n, n, -1)
    if interchain:
        z = z * (token_asym_id[:, :, None] == token_asym_id[:, None, :])[..., None].to(z.dtype)
    return residual + z if residual is not None and not fuse else z
