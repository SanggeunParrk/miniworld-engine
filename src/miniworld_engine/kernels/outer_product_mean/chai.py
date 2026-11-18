"""Chai-1 OuterProductMean (``groups = 8``, d_hidden 64, product LayerNorm) on H100: fused CUDA passes (``cuda/chai.cu``); on A100 the same passes
(``cuda/chai_sm80``: mma.sync / cp.async, d_pair 256).

    prologue   LayerNorm of the MSA rows + both projections + mask -> a, b [8, S N, 8] (group-major)         ``cuda.chai_proj``
    products   O[g] = a_g^T b_g over the MSA rows: one cuBLAS batched GEMM                                  [8, N K, N K]
    epilogue   per pair: LayerNorm of the 512 products, out-projection (+ bias), + residual, one TMA + wgmma pass   ``cuda.chai_epilogue``

The product LayerNorm's affine folds into the out-projection: W' = W_out . gamma (columns), b' = b_out + W_out beta.
Backward: dy = dz W' and dW' = dz^T y, dbias = sum dz (cuBLAS); the LayerNorm backward scatters dy into the grouped layout (``cuda.chai_bwd``);
dA = b dO^T, dB = a dO (two batched GEMMs); the projection / LayerNorm backward of the MSA rows in torch.  dgamma = sum_o W dW', dbeta = W^T dbias.
"""

from __future__ import annotations

import functools
import os
from pathlib import Path

import torch

from miniworld_engine.kernels._compile import opaque

G, K, P, CM = 8, 8, 512, 64


def serves_dims(
    msa: torch.Tensor, groups: int, group_width: int, d_pair: int, *, mask: torch.Tensor | None = None, projection_bias: bool = False
) -> bool:
    if not msa.is_cuda or msa.dtype != torch.bfloat16 or msa.ndim != 4 or msa.shape[0] != 1:
        return False
    cap = torch.cuda.get_device_capability(msa.device)
    if cap not in ((9, 0), (8, 0), (12, 0)) or (
        cap != (9, 0) and d_pair % 256
    ):  # A100 / RTX 5090 (cuda/chai_sm80): d_pair a multiple of 256
        return False
    _, s, n, cm = msa.shape
    return (
        groups == G
        and group_width == K
        and cm == CM
        and n % 64 == 0
        and s > 0
        and d_pair % 128 == 0
        and not projection_bias
        and (mask is None or mask.shape == msa.shape[:3])
    )


def serves(module, msa: torch.Tensor, mask: torch.Tensor | None, *, training: bool = False, interchain: bool = False) -> bool:
    if training and interchain:
        return False
    return module.to_out.weight.shape[1] == P and serves_dims(
        msa, module.groups, module.group_width, module.to_out.weight.shape[0], mask=mask, projection_bias=module.to_left.bias is not None
    )


def _kernels(t: torch.Tensor):
    """The pass kernels of the card: ``cuda`` (``chai.cu``, sm_90a) or ``cuda.chai_sm80`` (A100), the same entry points."""
    if torch.cuda.get_device_capability(t.device) in ((8, 0), (12, 0)):  # A100, RTX 5090 (the sm_120a build of the same sources)
        from miniworld_engine.kernels.outer_product_mean.cuda import chai_sm80

        return chai_sm80
    from miniworld_engine.kernels.outer_product_mean import cuda

    return cuda


def _fold(wo, gam, bet, bo):
    """(W' [CZ, 512] bf16, b' [CZ] fp32): the product LayerNorm's gamma / beta folded into the out-projection."""
    w = wo.float()
    return (w * gam.float()[None, :]).to(torch.bfloat16).contiguous(), (bo.float() + w @ bet.float()).contiguous()


def _forms(lnw, lnb, wl, wr, gam, bet, wo, bo, infer: bool = False):
    """The kernel-dtype forms of the weights, made once while the weights are unchanged (`_capture.param_forms`): (LayerNorm gamma / beta fp32,
    [W_left; W_right] bf16, W' bf16, b' fp32). With fp32 master parameters this was a handful of cast kernels per call."""
    from miniworld_engine.kernels import _capture

    def build():
        wf, bf = _fold(wo, gam, bet, bo)
        return lnw.float().contiguous(), lnb.float().contiguous(), torch.cat([wl, wr]).to(torch.bfloat16).contiguous(), wf, bf

    # ``infer`` (the inference forward): a capture reads the entry its eager warm-up made (the 4 fold / cast kernels, ~10 us at L 128, leave
    # the graph); training keeps the forms of the weights as they are at each replay
    return _capture.param_forms("chai_forms", [lnw, lnb, wl, wr, gam, bet, wo, bo], build, capture_reads_eager=infer)


def _front(msa, mask, forms, eps, save):
    cuda = _kernels(msa)

    _, s, n, _ = msa.shape
    x = msa[0].reshape(s * n, CM).contiguous()
    mk = None
    if mask.numel():
        mk = mask[0].reshape(s * n).contiguous()
        if mk.dtype != torch.bool:
            mk = mk != 0
    lnw32, lnb32, w, _, _ = forms
    a, b, st1 = cuda.chai_proj(x, mk, lnw32, lnb32, w, eps, save)
    products = getattr(cuda, "chai_products", None)  # A100: the masked product skips the dead tokens
    if not save and products is None:
        from miniworld_engine.kernels.integrations import opm_fp8

        if f8_wanted(a, s, n):
            opm_fp8.GUARD_STATS["calls"] += 1
            return a, b, st1, f8_products(a, b, s, n, forms)
        if opm_fp8.POLICY is not None and s >= F8_MIN_S:  # (below the depth gate the bf16 route is the intended one: not a fallback)
            opm_fp8.GUARD_STATS["fallbacks"] += 1
    o = (
        products(a, b, mk, s, n) if products is not None else torch.bmm(a.view(G, s, n * K).transpose(1, 2), b.view(G, s, n * K))
    )  # [8, N K, N K]
    return a, b, st1, o


def _fwd_op_fake(args, eps, eps_prod, train):
    """Shapes of ``_fwd_op``: z [1, N, N, CZ] bf16; with ``train`` also a, b [8, S N, 8], y [N N, 512], the epilogue and prologue statistics."""
    msa, wo = args[0], args[8]
    _, s, n, _ = msa.shape
    bf = torch.bfloat16
    new = lambda *shape, dt=bf: msa.new_empty(shape, dtype=dt)
    if not train:
        return [new(1, n, n, wo.shape[0]), new(0), new(0), new(0), new(0, dt=torch.float32), new(0, dt=torch.float32)]
    return [
        new(1, n, n, wo.shape[0]),
        new(G, s * n, K),
        new(G, s * n, K),
        new(n * n, P),
        new(n * n, 2, dt=torch.float32),
        new(s * n, 2, dt=torch.float32),
    ]


@opaque(fake=_fwd_op_fake, name="outer_product_mean_chai_fwd")
def _fwd_op(args: list[torch.Tensor], eps: float, eps_prod: float, train: bool) -> list[torch.Tensor]:
    """The Chai-1 OPM forward: prologue, grouped products, fused epilogue. args = [msa, mask, lnw, lnb, wl, wr, gamma, beta, wo, bo, residual] (empty tensors for no mask / residual)."""
    cuda = _kernels(args[0])
    msa, mask, lnw, lnb, wl, wr, gam, bet, wo, bo, residual = args
    forms = _forms(lnw, lnb, wl, wr, gam, bet, wo, bo, infer=not train)
    a, b, st1, o = _front(msa, mask, forms, eps, train)
    wf, bf = forms[3], forms[4]
    z, y, st = cuda.chai_epilogue(o, wf, bf, residual if residual.numel() else None, train, eps_prod)
    if not train:
        return [
            z,
            msa.new_empty(0),
            msa.new_empty(0),
            msa.new_empty(0),
            msa.new_empty(0, dtype=torch.float32),
            msa.new_empty(0, dtype=torch.float32),
        ]
    return [z, a, b, y, st, st1]


def _bwd_op_fake(args, kept, dz, eps, eps_prod):
    """Shapes of ``_bwd_op``: one gradient per differentiable argument (msa, LayerNorm, projections, product norm, out-projection), in the argument's dtype."""
    return [torch.empty_like(args[i], dtype=args[i].dtype) for i in (0, 2, 3, 4, 5, 6, 7, 8, 9)]


@opaque(fake=_bwd_op_fake, name="outer_product_mean_chai_bwd")
def _bwd_op(args: list[torch.Tensor], kept: list[torch.Tensor], dz: torch.Tensor, eps: float, eps_prod: float) -> list[torch.Tensor]:
    """Gradients of ``_fwd_op`` (train): [dmsa, dlnw, dlnb, dwl, dwr, dgamma, dbeta, dwo, dbo] from dz and the saved a, b, y, statistics."""
    cuda = _kernels(args[0])

    msa, mask, lnw, lnb, wl, wr, gam, bet, wo, bo, _ = args
    a, b, y, st, st1 = kept
    f32 = torch.float32
    _, s, n, _ = msa.shape
    t = s * n
    cz = wo.shape[0]
    dz2 = dz.reshape(n * n, cz).contiguous()
    lnw32, lnb32, wlr, wf, _ = _forms(lnw, lnb, wl, wr, gam, bet, wo, bo)
    dy = torch.mm(dz2, wf)  # [N N, 512] = dz W'
    dbo = torch.mm(dz2.new_ones((1, n * n)), dz2, out_dtype=f32).reshape(-1)  # sum over the pairs
    dwp = torch.mm(dz2.t(), y, out_dtype=f32)  # dW' = dz^T y
    # the RTX 5090: fewer, fused small kernels (dWo / dgamma / dbeta in one, split-K dW, one-launch column sums, dx returned without a copy)
    rtx5090 = torch.cuda.get_device_capability(msa.device) == (12, 0) and wo.dtype in (torch.float32, torch.bfloat16) and t % 512 == 0
    if rtx5090:
        from miniworld_engine.kernels.pair_weighted_averaging.cuda import gemm_sm120

        dwo, dgam, dbet = gemm_sm120.chai_wo_grads(dwp, dbo, wo, gam, bet)
    else:
        w32 = wo.float()
        dwo = dwp * gam.float()[None, :] + dbo[:, None] * bet.float()[None, :]  # W' = W gamma, b' = b + W beta
        dgam = (w32 * dwp).sum(0)
        dbet = w32.t() @ dbo
    do = cuda.chai_bwd(dy, y, st, n)
    del dy
    bwd_products = getattr(cuda, "chai_bwd_products", None)  # A100 / RTX 5090: the card's batched GEMMs
    if bwd_products is not None:
        da, db = bwd_products(a, b, do, s, n)  # [8, S, N K]
    else:
        da = torch.bmm(b.view(G, s, n * K), do.transpose(1, 2))  # [8, S, N K]
        db = torch.bmm(a.view(G, s, n * K), do)
    del do
    x = msa[0].reshape(t, CM).contiguous()
    mk = None
    if mask.numel():
        mk = mask[0].reshape(t).contiguous()
        if mk.dtype != torch.bool:
            mk = mk != 0
    dcat, ln16, dx, part = cuda.chai_projbwd(x, mk, da.view(G, t, K), db.view(G, t, K), lnw32, lnb32, st1, wlr)
    if rtx5090:  # dW = dcat^T ln16 as 64 K-chunk partials (64 x 64 tiles) summed into [W_left | W_right] directly
        dwl, dwr = gemm_sm120.colsum2(gemm_sm120.wgrad_parts(dcat, ln16, 5, 64), G * K * CM)
        dlw, dlb = gemm_sm120.colsum2(part, CM)
        return [
            dx.view(1, s, n, CM),
            dlw.to(lnw.dtype),
            dlb.to(lnb.dtype),
            dwl.view(G * K, CM).to(wl.dtype),
            dwr.view(G * K, CM).to(wr.dtype),
            dgam.to(gam.dtype),
            dbet.to(bet.dtype),
            dwo.to(wo.dtype),
            dbo.to(bo.dtype),
        ]
    dw = torch.mm(dcat.t(), ln16, out_dtype=f32)  # [128, 64]
    dlw, dlb = part[:, :CM].sum(0), part[:, CM:].sum(0)
    return [
        dx.view(1, s, n, CM).clone(),
        dlw.to(lnw.dtype),
        dlb.to(lnb.dtype),
        dw[: G * K].to(wl.dtype).clone(),
        dw[G * K :].to(wr.dtype).clone(),
        dgam.to(gam.dtype),
        dbet.to(bet.dtype),
        dwo.to(wo.dtype),
        dbo.to(bo.dtype),
    ]


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
    args = [
        msa.contiguous(),
        empty if mask is None else mask,
        module.ln_msa.weight,
        module.ln_msa.bias,
        module.to_left.weight,
        module.to_right.weight,
        module.product_norm.weight,
        module.product_norm.bias,
        module.to_out.weight,
        module.to_out.bias,
        residual.contiguous() if fuse and residual is not None else empty,
    ]
    eps, eps_prod = float(module.ln_msa.eps), float(module.product_norm.eps)
    z = _ChaiFn.apply(eps, eps_prod, *args) if torch.is_grad_enabled() else _fwd_op(args, eps, eps_prod, False)[0]
    z = z.view(1, n, n, -1)
    if interchain and token_asym_id is not None:
        z = z * (token_asym_id[:, :, None] == token_asym_id[:, None, :])[..., None].to(z.dtype)
    return residual + z if residual is not None and not fuse else z


# ---- EXPERIMENTAL e4m3 grouped product (opt-in with ``MINIWORLD_OPM_FP8=bound``, H100 inference; docs/fp8.md "Chai-1 grouped OPM") ----
#: the shortest MSA depth S (padded rows of the call) that takes the e4m3 product; below it the bf16 batched GEMM runs unchanged
# default: never (measured a net loss at every S, docs/fp8.md); 0 = every depth
F8_MIN_S = int(os.environ.get("MINIWORLD_OPM_FP8_CHAI_MINS", str(1 << 40)))


@functools.cache
def _f8_ext():
    from miniworld_engine.kernels._nvcc import ensure_cuda_home, gencodes, host_flags, load_extension

    ensure_cuda_home()
    return load_extension(
        name="opm_chai_f8",
        sources=[str(Path(__file__).parent / "cuda" / "chai_f8.cu")],
        extra_cuda_cflags=[*host_flags(), "-O3", "-std=c++17", *gencodes("90a")],
        extra_cflags=["-std=c++17"],
        verbose=False,
    )


def _f8_forms(forms):
    """(sa, sb, 1 / sa, 1 / sb): the weight-bound power-of-two dequantisation scales of a and b (``opm_fp8.bound_scales``: |LN(x)_k| <= sqrt(63) |gamma_k| + |beta_k|
    for every input bounds every a / b value, so the cast cannot saturate), fp32 [1] device tensors, made once per weight pack."""
    from miniworld_engine.kernels import _capture
    from miniworld_engine.kernels.integrations import opm_fp8

    lnw, lnb, w, _, _ = forms

    def build():
        s = opm_fp8.bound_scales(lnw, lnb, w[: G * K], w[G * K :])
        # own allocations: cuBLASLt wants aligned scale pointers
        return s[0:1].clone(), s[1:2].clone(), (1.0 / s[0:1]).clone(), (1.0 / s[1:2]).clone()

    return _capture.param_forms("chai_f8_scales", [lnw, lnb, w], build, capture_reads_eager=True)


def f8_wanted(a: torch.Tensor, s: int, n: int) -> bool:
    """Whether the call takes the e4m3 product: the switch is on (``opm_fp8.POLICY == "bound"``: the weight-bound scales need no calibration or guard), H100, inference, the shape fits cuBLASLt's e4m3 TN form."""
    from miniworld_engine.kernels.integrations import opm_fp8

    if opm_fp8.POLICY != "bound" or torch.cuda.get_device_capability(a.device) != (9, 0) or s < F8_MIN_S:
        return False
    return s % 16 == 0 and (n * K) % 16 == 0


def f8_products(a: torch.Tensor, b: torch.Tensor, s: int, n: int, forms) -> torch.Tensor:
    """O [8, N K, N K] bf16 = the e4m3 batched product of a, b [8, S N, 8] (the same value the bf16 ``bmm`` forms, up to the e4m3 rounding of its operands)."""
    ext = _f8_ext()
    sa, sb, ia, ib = _f8_forms(forms)
    a8 = ext.chai_quant_t(a.view(G, s, n * K), ia)
    b8 = ext.chai_quant_t(b.view(G, s, n * K), ib)
    o = torch.empty((G, n * K, n * K), device=a.device, dtype=torch.bfloat16)
    s0, s1 = sa.view(()), sb.view(())
    for g in range(G):
        o[g] = torch._scaled_mm(a8[g], b8[g].t(), s0, s1, out_dtype=torch.bfloat16)  # pyright: ignore[reportPrivateImportUsage]
    return o
