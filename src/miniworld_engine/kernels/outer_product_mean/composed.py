"""H100 inference of the OuterProductMean variants the fused OPM kernels do not take (``modules.outer_product.OuterProductMean``), one
group (``groups = 1``) of K = 32 channels:

    RoseTTAFold3   left / right projections with a bias (``projection_bias``): masked rows lose the bias too

    m = LN(msa); L, R = mask (m W^T [+ b]) (cuBLAS); prod = L^T R over the MSA rows (one batched GEMM, [N K, N K] per b); the rows (i, j)
    of the K K products (``cuda.gather``); the out-projection (cuBLAS), and the division by the pair counts (clamped at 1) before or after
    it (``normalize_before_proj``).

``count_offset`` (divisor ``count + offset``) changes only the divisor: ``opm_train.divisor_params`` gives
(dmin, doff) with divisor = max(count, dmin) + doff, which every count consumer below takes (the epilogues).
"""

from __future__ import annotations

import torch


def serves(module, msa: torch.Tensor, mask: torch.Tensor | None) -> bool:
    if torch.is_grad_enabled() or not msa.is_cuda or msa.dtype != torch.bfloat16 or msa.ndim != 4:
        return False
    if torch.cuda.get_device_capability(msa.device) != (9, 0):
        return False
    g, k = module.groups, module.group_width
    return g == 1 and k % 8 == 0 and g * k * k in (512, 1024) and (mask is None or mask.shape == msa.shape[:3])


def inference(module, msa, mask, token_asym_id=None, residual=None):
    from miniworld_engine.kernels.integrations import opm_train
    from miniworld_engine.kernels.outer_product_mean import cuda

    B, S, N, _ = msa.shape
    G, K = module.groups, module.group_width
    m = module.ln_msa(msa).reshape(B * S * N, -1)
    keep = None if mask is None else mask.reshape(B * S * N, 1).to(m.dtype)
    if keep is not None and module.to_left.bias is None:  # mask (W m) = W (mask m)
        m = m * keep

    def project(linear):
        # one GEMM (the bias in its epilogue): [B, S, N K], so the outer product below reads [N K, S] / [S, N K] views without a copy
        out = torch.mm(m, linear.weight.t()) if linear.bias is None else torch.addmm(linear.bias, m, linear.weight.t())
        if linear.bias is not None and keep is not None:
            out = out * keep
        return out.view(B, S, N * K)

    left, right = project(module.to_left), project(module.to_right)
    prod = torch.bmm(left.transpose(1, 2), right)  # [b, (i k), (j l)]
    rows = cuda.gather(prod, G, K).view(B, N, N, K * K)
    mf = (mask if mask is not None else torch.ones(B, S, N, dtype=torch.bool, device=msa.device)).float()
    dmin, doff = opm_train.divisor_params(module, S)
    count = (torch.einsum("bsi,bsj->bij", mf, mf).clamp(min=dmin) + doff)[..., None]
    # the count division before or after the out-projection (``normalize_before_proj``)
    pair = module.to_out((rows / count).to(rows.dtype)) if module.normalize_before_proj else (module.to_out(rows) / count).to(rows.dtype)
    if module.mask_interchain and token_asym_id is not None:
        pair = pair * (token_asym_id[:, :, None] == token_asym_id[:, None, :])[..., None].to(pair.dtype)
    return residual + pair if residual is not None else pair


def biased_wide_serves(module, msa, mask) -> bool:
    """The wide H100 composition takes the biases in its prologue: one sample, S % 256, N % 64, d_msa 64 / 128, 32 + 32 projections."""
    if torch.is_grad_enabled() or not msa.is_cuda or msa.dtype != torch.bfloat16 or msa.ndim != 4 or msa.shape[0] != 1:
        return False
    if torch.cuda.get_device_capability(msa.device) != (9, 0) or module.groups != 1 or module.group_width != 32:
        return False
    _, s, n, cm = msa.shape
    return s % 256 == 0 and n % 64 == 0 and cm in (64, 128) and module.to_out.weight.shape[0] in (128, 256, 384)


def biased_wide(module, msa, mask, token_asym_id=None, residual=None):
    """RoseTTAFold3's biased projections in the wide H100 composition: its persistent TMA prologue adds the left | right biases in
    the projection epilogue (masked like the projections), then cuBLAS for the outer product and the fused epilogue (out-projection,
    bias, count normalisation before or after it, residual)."""
    from miniworld_engine.kernels.integrations import opm_train

    bf = torch.bfloat16
    m = msa[0].contiguous()
    _, n, cm = m.shape
    cz = module.to_out.weight.shape[0]
    fused = opm_train._fused_enabled(m.shape[0], n, training=False, cm=cm, cz=cz) and torch.cuda.get_device_capability(msa.device) == (9, 0)
    ext = opm_train._ext(msa.device, native=True, tma=False, cm=cm, cz=cz, fused=fused)
    dmin, doff = opm_train.divisor_params(module, msa.shape[1])
    mk = (mask[0] if mask is not None else torch.ones(m.shape[:2], dtype=torch.bool, device=msa.device)).contiguous()
    if mk.dtype != torch.bool:
        mk = mk != 0
    from miniworld_engine.kernels import _capture

    params = [
        module.ln_msa.weight,
        module.ln_msa.bias,
        module.to_left.weight,
        module.to_right.weight,
        module.to_out.weight,
        module.to_out.bias,
        module.to_left.bias,
        module.to_right.bias,
    ]
    if cm == 64 and cz == 128 and all(t.is_contiguous() and t.dtype in (torch.float32, bf) for t in params):
        # one packing launch for the LayerNorm affine, the bf16 weights, the rounded bias and the projection biases (was six casts and a cat),
        # made once while the weights are unchanged
        def packed():
            gam, bet, wa16, wb16, wo16, ob, _, _, pbias = ext.opm_native_pack(params[:6], True, params[6], params[7])
            return gam, bet, wa16, wb16, wo16, ob, pbias

        forms = packed
    else:

        def converted():
            return (
                module.ln_msa.weight.float().contiguous(),
                module.ln_msa.bias.float().contiguous(),
                module.to_left.weight.to(bf).contiguous(),
                module.to_right.weight.to(bf).contiguous(),
                module.to_out.weight.to(bf).contiguous(),
                module.to_out.bias.to(bf).float().contiguous(),
                torch.cat([module.to_left.bias, module.to_right.bias]).float().contiguous(),
            )

        forms = converted
    gam, bet, wa16, wb16, wo_epi, bo_epi, pbias = _capture.param_forms("opm_biased_wide", params, forms)
    # the epilogue takes the mask count from the mask bit rows (`norm` is that int32 [N, S/32] table): no count kernel
    dkw = {} if (dmin, doff) == (1.0, 0.0) else {"dmin": dmin, "doff": doff}  # the pair divisor max(count, dmin) + doff
    a2, bt, _, _, norm = ext.opm_prologue_wide(m, mk, gam, bet, wa16, wb16, float(getattr(module.ln_msa, "eps", 1e-5)), False, pbias, False)
    interchain = module.mask_interchain and token_asym_id is not None
    res = None if interchain else residual
    if fused:  # the products stay in shared memory (norm is the mask bit table here)
        z = ext.opm_fused_infer(
            a2, bt, norm, opm_train._fused_weight(module.to_out.weight), bo_epi, res, int(not module.normalize_before_proj), **dkw
        )
    else:
        z = ext.opm_epilogue(torch.mm(a2, bt.t()), norm, wo_epi, bo_epi, n, n, res, int(not module.normalize_before_proj), **dkw)
    z = z.view(1, n, n, cz)
    if interchain and token_asym_id is not None:
        z = z * (token_asym_id[:, :, None] == token_asym_id[:, None, :])[..., None].to(z.dtype)
        return residual + z if residual is not None else z
    return z
