"""H100 bf16 inference of the ATOM DiT block variants (the Boltz-2, RoseTTAFold3 and Chai-1 atom DiT blocks, ``modules.atom_dit``).

The token blocks (TokenDiTBoltz2 / RoseTTAFold3 / Chai1) are not here: they run the one token DiT algorithm, TokenDiTProtenix's ``"cuda"``
route (``kernels/integrations/token_dit`` / ``token_dit_train`` with the model's options).

Per step, for the whole stack:
  * the conditioning tables: ONE GEMM for every block's AdaLN scale / shift (the weight-only cond LayerNorm folded into the
    weights, Chai-1's ``+ 1`` into the bias) and ONE for every block's output gates (the conditioning is shared by the samples);
  * the pair bias of every block (``_atom_biases``), per pair / mask / uid (``_capture.lookup_inputs``: computed once while the pair is
    unchanged).
Per block: CUDA rows (``cuda/rows.cu``: LN + AdaLN, residual + gates, kq LayerNorm, (up-)GLU), cuBLAS products, and the local DiT's
window attention.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from miniworld_engine.kernels import _capture
from miniworld_engine.kernels.dit_variants.cuda.infer32 import _atom_biases, _static_pack

# ------------------------------------------------------------------------------------------------------------------------- atom
# The atom stack: the conditioning tables as for the token stack (per atom, [N, nb 6 128] and [N, nb 2 128]); the pair bias of every
# block from the local DiT's ``pair_bias_mma`` (LN gamma . W log2(e), -inf at keys outside [0, N) or masked), Chai-1's same-conformer
# mask folded in as -inf; per block the CUDA rows at 128 wide, cuBLAS, and the local DiT's window attention (``attn_fwd_mma``: the
# clipped key window; the slid one's edge windows are redone in torch, as the AF3 local DiT does).

_ATOM_PACKS: dict = {}
_ATOM_BIASES: dict = {}


def atom_serves(stack, single, cond, pair, mask, uid) -> bool:
    if torch.is_grad_enabled() or not single.is_cuda or single.dtype is not torch.bfloat16 or single.ndim != 3:
        return False
    if torch.cuda.get_device_capability(single.device) != (9, 0):
        return False
    _, n, d = single.shape
    shared = cond.ndim == 2 or (cond.ndim == 3 and (cond.shape[0] == 1 or cond.stride(0) == 0))
    return (
        shared
        and d == 128
        and cond.shape[-2:] == (n, 128)
        and n % 32 == 0
        and tuple(pair.shape[:3]) == (n // 32, 32, 128)
        and (mask is None or tuple(mask.shape) == (n,))
        and (uid is None or tuple(uid.shape) == (n,))
        and stack.blocks[0].expand.weight.shape[0] == 4 * d
    )


def _atom_pack(stack):
    bf = torch.bfloat16
    blocks = list(stack.blocks)
    mod_w, mod_b, gate_w, gate_b, per = [], [], [], [], []
    for i, blk in enumerate(blocks):
        for norm in (blk.q_adaln, blk.k_adaln, blk.trans_adaln):
            dev = norm.to_scale.weight.device
            if norm.identity_scale:
                mod_w += [norm.to_scale.weight.float(), norm.to_shift.weight.float()]
                mod_b += [torch.ones(128, device=dev), torch.zeros(128, device=dev)]
            else:
                w = norm.cond_norm_weight.float()[None, :]
                mod_w += [norm.to_scale.weight.float() * w, norm.to_shift.weight.float() * w]
                mod_b += [norm.to_scale.bias.float(), torch.zeros(128, device=dev)]
        gate_w += [blk.attn_zero.weight.float(), blk.trans_zero.weight.float()]
        gate_b += [blk.attn_zero.bias.float(), blk.trans_zero.bias.float()]
        if stack.per_block_pair_norm:
            gamma, wb = blk.pair_norm.weight, blk.to_pair_bias.weight
        else:
            gamma, wb = stack.pair_norm.weight, stack.to_pair_bias.weight[4 * i:4 * i + 4]
        entry = {
            "wq": blk.to_q.weight.to(bf).contiguous(), "bq": blk.to_q.bias.to(bf).contiguous(),
            "wkv": torch.cat([blk.to_k.weight, blk.to_v.weight]).to(bf).contiguous(),
            "wexp": torch.cat([blk.expand.weight, *((blk.up.weight,) if blk.up_gate else ())]).to(bf).contiguous(),
            "wsq": blk.squeeze.weight.to(bf).contiguous(),
            "gamma": gamma.float().contiguous(), "wb": wb.float().contiguous(),
        }
        if blk.gating:
            entry["wg"] = blk.to_g.weight.to(bf).contiguous()
        if blk.project_output:
            entry["wo"] = blk.to_out.weight.to(bf).contiguous()
        if blk.kq_norm:
            entry["qk"] = tuple(t.float().contiguous() for t in (blk.q_norm.weight, blk.q_norm.bias, blk.k_norm.weight, blk.k_norm.bias))
            entry["qk_eps"] = float(blk.q_norm.eps)
        per.append(entry)
    return {
        "mod_w": torch.cat(mod_w).to(bf).contiguous(), "mod_b": torch.cat(mod_b).to(bf).contiguous(),
        "gate_w": torch.cat(gate_w).to(bf).contiguous(), "gate_b": torch.cat(gate_b).to(bf).contiguous(), "blocks": per,
    }


def _atom_edges(o, q, k, v, edge, a, n):
    """The slid key window's edge windows (``_atom_biases``) redone in torch (fp32) over their keys, written over ``o``."""
    widx, keys, bias = edge
    h, d = 4, 32
    nwin = n // 32
    qs = q.view(a, nwin, 32, h, d)[:, widx].float()
    ks = k.view(a, n, h, d)[:, keys].float()
    vs = v.view(a, n, h, d)[:, keys].float()
    logits = torch.einsum("awqhd,awkhd->awhqk", qs, ks) * d**-0.5 + bias[None]
    out = torch.einsum("awhqk,awkhd->awqhd", torch.nan_to_num(logits.softmax(-1)), vs).reshape(a, 5, 32, 128)
    o.view(a, nwin, 32, 128).index_copy_(1, widx, out.to(o.dtype))


def atom_inference(stack, single, cond, pair, mask, uid):
    from miniworld_engine.kernels.augmented_attention.cuda import (
        sm100_atom_local as local,
    )
    from miniworld_engine.kernels.dit_variants import cuda

    bf = torch.bfloat16
    blocks = list(stack.blocks)
    nb = len(blocks)
    b0 = blocks[0]
    a, n0, _ = single.shape
    uid_in = uid
    n = (n0 + 63) // 64 * 64        # the window kernel takes 64-query tiles: one padding window, masked, when N is an odd number of 32
    kmask = (torch.arange(n0, device=single.device) < n0) if mask is None else mask.bool()
    c = (cond if cond.ndim == 2 else cond[0]).to(bf)
    z = pair.to(bf)
    if n != n0:
        single = F.pad(single, (0, 0, 0, n - n0))
        c = F.pad(c, (0, 0, 0, n - n0))
        z = F.pad(z, (0, 0, 0, 0, 0, 0, 0, (n - n0) // 32))
        kmask = F.pad(kmask, (0, n - n0), value=False)
        uid = None if uid is None else F.pad(uid, (0, n - n0), value=-1)
    c, z, kmask = c.contiguous(), z.contiguous(), kmask.contiguous()
    pack = _static_pack(_ATOM_PACKS, stack, _atom_pack)
    sig = not b0.q_adaln.identity_scale
    sm = F.layer_norm(c.float(), (128,), eps=1e-5).to(bf) if sig else c
    t_mod = torch.addmm(pack["mod_b"], sm, pack["mod_w"].t())      # [N, nb 6 128]: per block q sc | sh, k sc | sh, t sc | sh
    t_gate = torch.addmm(pack["gate_b"], c, pack["gate_w"].t())    # [N, nb 2 128]: per block gA | gT
    pair_eps = (b0.pair_norm if stack.per_block_pair_norm else stack.pair_norm).eps
    # keyed by the caller's pair / mask / uid tensors (kmask and a padded uid are made per call)
    key = (pair.data_ptr(), pair._version, tuple(pair.shape), None if mask is None else (mask.data_ptr(), mask._version),
           None if uid_in is None else (uid_in.data_ptr(), uid_in._version), id(pack), stack.key_window)
    slide = stack.key_window == "slide"
    biases = _capture.lookup_inputs(  # the entry keeps its key tensors alive
        _ATOM_BIASES, key, lambda: (_atom_biases(pack, z, kmask, uid, n, pair_eps, slide), pair, mask, uid_in), limit=2)[0]

    m = a * n
    dev = single.device
    x = torch.empty(m, 128, device=dev, dtype=torch.float32)
    xq, xk = torch.empty(m, 128, device=dev, dtype=bf), torch.empty(m, 128, device=dev, dtype=bf)
    xt = torch.empty_like(xq)
    kv = torch.empty(m, 256, device=dev, dtype=bf)
    width = b0.expand.weight.shape[0] // 2
    ab = torch.empty(m, (3 if b0.up_gate else 2) * width, device=dev, dtype=bf)
    h = torch.empty(m, width, device=dev, dtype=bf)
    out = torch.empty(m, 128, device=dev, dtype=bf)
    eps = b0.adaln_eps
    for i, (blk, w) in enumerate(zip(blocks, pack["blocks"], strict=True)):
        t = t_mod[:, 6 * 128 * i:6 * 128 * (i + 1)]
        ms = [t[:, 128 * j:128 * (j + 1)] for j in range(6)]
        ga, gt = t_gate[:, 256 * i:256 * i + 128], t_gate[:, 256 * i + 128:256 * (i + 1)]
        last = out if i == nb - 1 else None
        mods = [(ms[0], ms[1], xq), (ms[2], ms[3], xk), *([(ms[4], ms[5], xt)] if stack.parallel else [])]
        cuda.ln_mod_rows(x, mods, n, eps, sig, xin=single.reshape(m, 128).contiguous() if i == 0 else None)
        q = torch.addmm(w["bq"], xq, w["wq"].t())
        torch.mm(xk, w["wkv"].t(), out=kv)
        k, v = kv[:, :128].contiguous(), kv[:, 128:].contiguous()
        if blk.kq_norm:
            cuda.qk_ln_rows(q, k, *w["qk"], w["qk_eps"])
        bias, edge = biases[i]
        o, _ = local.attn_fwd_mma(q.view(a, n, 128), k.view(a, n, 128), v.view(a, n, 128), bias)
        if stack.key_window == "slide":
            _atom_edges(o, q, k, v, edge, a, n)
        o = o.view(m, 128)
        if blk.gating:
            cuda.gate_rows_inplace(o, torch.mm(xq, w["wg"].t()))
        ya = torch.mm(o, w["wo"].t()) if blk.project_output else o
        if stack.parallel:
            torch.mm(xt, w["wexp"].t(), out=ab)
            cuda.glu_rows(ab, h, blk.up_gate)
            cuda.res_rows(x, ya, ga, n, yt=torch.mm(h, w["wsq"].t()), gt=gt, final=last)
        else:
            cuda.res_rows(x, ya, ga, n, ms=ms[4], mb=ms[5], out_ln=xt, eps=eps, sig=sig)
            torch.mm(xt, w["wexp"].t(), out=ab)
            cuda.glu_rows(ab, h, blk.up_gate)
            cuda.res_rows(x, torch.mm(h, w["wsq"].t()), gt, n, final=last)
    return out.view(a, n, 128)[:, :n0]
