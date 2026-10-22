"""H100 inference of the DiT block variants (``modules.dit_variant``): the AF3 token step's structure with the variants' changes.

Per step, for the whole stack:
  * the conditioning tables: ONE GEMM for every block's AdaLN scale / shift (the weight-only cond LayerNorm folded into the
    weights, Chai-1's ``+ 1`` into the bias) and ONE for every block's output gates, [L, *] bf16 (the conditioning is shared by
    the samples);
  * the pair bias of every block: one LayerNorm of the pair rows (each block's LN weight folded into its projection; an LN
    bias only shifts a head's logits by a constant, which the softmax cancels) and one GEMM, [nb 16, L, L] bf16 in log2 units,
    masked keys -inf -- per pair / mask (``_capture.lookup_inputs``: computed once while the pair is unchanged).
Per block: CUDA rows (``cuda/rows.cu``: LN + AdaLN, residual + gates, kq LayerNorm, (up-)GLU), cuBLAS products, and the AF3 token
step's attention core (``augmented_attention.cuda.gated_inference``, q in exp2 units: the logit scale is folded into W_q / b_q,
or into the kq-norm's affine). A block without a gating query gets a gate column of +1e4 (sigmoid = 1 exactly).
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from miniworld_engine.kernels import _capture

LOG2E = 1.4426950408889634
#: gate logits of a block without a gating query: sigmoid(1e4) == 1.0 in fp32
_NO_GATE = 1.0e4
_PACKS: dict = {}
_BIASES: dict = {}


def serves(stack, single, cond, pair, mask) -> bool:
    """A bf16 call without a gradient on H100 at the AF3 token core's layout (768 wide, 16 x 48 heads, L a multiple of 128),
    the conditioning shared by the samples."""
    if torch.is_grad_enabled() or not single.is_cuda or single.dtype is not torch.bfloat16:
        return False
    if torch.cuda.get_device_capability(single.device) != (9, 0):
        return False
    from miniworld_engine.kernels.augmented_attention.cuda import inference_core_supported

    blk = stack.blocks[0]
    if single.ndim != 3:
        return False
    _, L, D = single.shape
    shared = cond.ndim == 2 or (cond.ndim == 3 and (cond.shape[0] == 1 or cond.stride(0) == 0))
    return (
        shared
        and cond.shape[-2] == L
        and tuple(pair.shape[:2]) == (L, L)
        and pair.shape[-1] in (128, 256)
        and (mask is None or tuple(mask.shape) == (L,))
        and blk.expand.weight.shape[0] == 4 * D
        and inference_core_supported(torch.bfloat16, L, D, blk.n_head, single.device.index or 0)
    )


def _pack(stack):
    """bf16 weights of the step, folded (see the module docstring)."""
    bf = torch.bfloat16
    blocks = list(stack.blocks)
    b0 = blocks[0]
    D = b0.to_q.weight.shape[0]
    qs = LOG2E / math.sqrt(D // b0.n_head)
    mod_w, mod_b, gate_w, gate_b, pair_w, per = [], [], [], [], [], []
    for blk in blocks:
        for norm in (blk.attn_adaln, blk.trans_adaln):
            if norm.identity_scale:
                mod_w += [norm.to_scale.weight.float(), norm.to_shift.weight.float()]
                mod_b += [torch.ones(D, device=norm.to_scale.weight.device), torch.zeros(D, device=norm.to_scale.weight.device)]
            else:
                w = norm.cond_norm_weight.float()[None, :]
                mod_w += [norm.to_scale.weight.float() * w, norm.to_shift.weight.float() * w]
                mod_b += [norm.to_scale.bias.float(), torch.zeros_like(norm.to_scale.bias, dtype=torch.float32)]
        gate_w += [blk.attn_zero.weight.float(), blk.trans_zero.weight.float()]
        gate_b += [blk.attn_zero.bias.float(), blk.trans_zero.bias.float()]
        pair_w.append(blk.to_pair_bias.weight.float() * blk.pair_norm.weight.float()[None, :] * LOG2E)
        q_scale = 1.0 if blk.kq_norm else qs
        rows = [blk.to_q.weight.float() * q_scale, blk.to_k.weight.float(), blk.to_v.weight.float()]
        bias = [blk.to_q.bias.float() * q_scale, torch.zeros(2 * D, device=blk.to_q.bias.device)]
        if blk.gating:
            rows.append(blk.to_g.weight.float())
            bias.append(torch.zeros(D, device=blk.to_q.bias.device))
        entry = {
            "wqkv": torch.cat(rows).to(bf).contiguous(),
            "bqkv": torch.cat(bias).to(bf).contiguous(),
            "wo": blk.to_out.weight.to(bf).contiguous(),
            "wexp": torch.cat([blk.expand.weight, *((blk.up.weight,) if blk.up_gate else ())]).to(bf).contiguous(),
            "wsq": blk.squeeze.weight.to(bf).contiguous(),
        }
        if blk.kq_norm:
            entry["qk"] = (
                (blk.q_norm.weight.float() * qs).contiguous(), (blk.q_norm.bias.float() * qs).contiguous(),
                blk.k_norm.weight.float().contiguous(), blk.k_norm.bias.float().contiguous(), float(blk.q_norm.eps))
        per.append(entry)
    return {
        "mod_w": torch.cat(mod_w).to(bf).contiguous(), "mod_b": torch.cat(mod_b).to(bf).contiguous(),
        "gate_w": torch.cat(gate_w).to(bf).contiguous(), "gate_b": torch.cat(gate_b).to(bf).contiguous(),
        "pair_w": torch.cat(pair_w).to(bf).contiguous(), "blocks": per,
    }


def _bias_all(pack, pair, mask, eps, heads_total):
    from miniworld_engine.kernels.dit_variants import cuda

    L, C = pair.shape[0], pair.shape[-1]
    zn = torch.empty(L * L, C, device=pair.device, dtype=torch.bfloat16)
    cuda.ln_rows(pair.reshape(L * L, C).to(torch.bfloat16).contiguous(), zn, eps)
    bias = torch.mm(pack["pair_w"], zn.t()).view(heads_total, L, L)
    if mask is not None:
        bias.masked_fill_(~mask.bool()[None, None, :], float("-inf"))
    return bias, pair, mask  # the entry keeps its key tensors alive


def inference(stack, single, cond, pair, mask):
    from miniworld_engine.kernels.augmented_attention import cuda as attention
    from miniworld_engine.kernels.dit_variants import cuda

    blocks = list(stack.blocks)
    b0 = blocks[0]
    nb, H = len(blocks), b0.n_head
    A, L, D = single.shape
    M = A * L
    params = list(stack.parameters())
    pack = _capture.lookup(_PACKS, (id(stack), *((p.data_ptr(), p._version) for p in params)), lambda: _pack(stack), limit=8)
    bf = torch.bfloat16
    s = (cond if cond.ndim == 2 else cond[0]).to(bf).contiguous()
    sig = not b0.attn_adaln.identity_scale
    sm = F.layer_norm(s.float(), s.shape[-1:], eps=1e-5).to(bf) if sig else s
    t_mod = torch.addmm(pack["mod_b"], sm, pack["mod_w"].t())     # [L, nb 4 D]: per block scA | shA | scT | shT
    t_gate = torch.addmm(pack["gate_b"], s, pack["gate_w"].t())   # [L, nb 2 D]: per block gA | gT
    key = (pair.data_ptr(), pair._version, tuple(pair.shape), pair.stride(),
           None if mask is None else (mask.data_ptr(), mask._version), pack["pair_w"].data_ptr())
    bias = _capture.lookup_inputs(_BIASES, key, lambda: _bias_all(pack, pair, mask, b0.pair_norm.eps, nb * H), limit=2)[0]

    x = torch.empty(M, D, device=single.device, dtype=torch.float32)
    xa = torch.empty(M, D, device=single.device, dtype=bf)
    xt = torch.empty_like(xa) if stack.parallel else xa
    qkvg = torch.empty(M, 4 * D, device=single.device, dtype=bf)
    if not b0.gating:
        qkvg[:, 3 * D:].fill_(_NO_GATE)
    width = b0.expand.weight.shape[0] // 2
    ab = torch.empty(M, (3 if b0.up_gate else 2) * width, device=single.device, dtype=bf)
    h = torch.empty(M, width, device=single.device, dtype=bf)
    out = torch.empty(M, D, device=single.device, dtype=bf)
    eps = b0.adaln_eps
    for i, (blk, w) in enumerate(zip(blocks, pack["blocks"], strict=True)):
        o = 4 * i * D
        msa, mba, mst, mbt = (t_mod[:, o + j * D:o + (j + 1) * D] for j in range(4))
        ga, gt = t_gate[:, 2 * i * D:(2 * i + 1) * D], t_gate[:, (2 * i + 1) * D:(2 * i + 2) * D]
        last = out if i == nb - 1 else None
        two = {"ms2": mst, "mb2": mbt, "out2": xt} if stack.parallel else {}
        cuda.ln_mod_rows(x, msa, mba, xa, L, eps, sig, xin=single.reshape(M, D) if i == 0 else None, **two)
        torch.addmm(w["bqkv"], xa, w["wqkv"].t(), out=qkvg if blk.gating else qkvg[:, :3 * D])
        if blk.kq_norm:
            wq, bq, wk, bk, qeps = w["qk"]
            cuda.qk_ln_rows(qkvg, wq, bq, wk, bk, qeps)
        attention.gated_inference(qkvg, bias, i, A, H, D // H)
        ya = torch.mm(qkvg[:, :D], w["wo"].t())
        if stack.parallel:
            torch.mm(xt, w["wexp"].t(), out=ab)
            cuda.glu_rows(ab, h, blk.up_gate)
            cuda.res_rows(x, ya, ga, L, yt=torch.mm(h, w["wsq"].t()), gt=gt, final=last)
        else:
            cuda.res_rows(x, ya, ga, L, ms=mst, mb=mbt, out_ln=xt, eps=eps, sig=sig)
            torch.mm(xt, w["wexp"].t(), out=ab)
            cuda.glu_rows(ab, h, blk.up_gate)
            cuda.res_rows(x, torch.mm(h, w["wsq"].t()), gt, L, final=last)
    return out.view(A, L, D)
