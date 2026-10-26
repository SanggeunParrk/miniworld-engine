"""H100 fp32 (TF32) INFERENCE of the DiT block variants (Boltz-2, RoseTTAFold3, Chai-1; token and atom stacks): the training rows (``train_rows.cu``,
``edge.cu``) and cuBLAS TF32 GEMMs, without anything the backward would need, and with everything that does not depend on the sample rows hoisted:

  * the folded fp32 weights of the stack, made once per (parameter pointer, version) (``interface._static_pack``'s rule);
  * the conditioning tables of EVERY block from two GEMMs per step (the weight-only LayerNorm of the sigmoid form folded into the weights, Chai-1's
    ``+ 1`` into the bias); one ``[L, 6 D]`` / ``[N, 8 D]`` slice per block;
  * the attention bias of every block: one pair LayerNorm + one stacked GEMM (token, key-permuted for the TF32 core, masked keys -inf), or the local
    DiT's ``pair_bias_mma`` + edge tables (atom), per pair / mask (``_capture.lookup_inputs``).
Per block: ``ln_mod_fwd``, cuBLAS q | k (| g) and v^T, the TF32 attention core (token: the gated inference core, g folded into its epilogue; atom: the
window kernel + the slid edge windows), ``res_*`` rows, cuBLAS expand, ``glu_fwd``, cuBLAS squeeze. bf16 inference is ``interface.py``'s.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from miniworld_engine import settings
from miniworld_engine.kernels import _capture
from miniworld_engine.kernels.dit_variants import interface as _bf16

LOG2E = 1.4426950408889634
_NO_GATE = 1.0e4
_PACKS: dict = {}
_TOKEN_BIASES: dict = {}
_ATOM_BIASES: dict = {}
# column blocks (units of D) of a block's table slice: [mod ..., gA, gT]
T_SC_A, T_SH_A, T_SC_T, T_SH_T, T_G_A, T_G_T = range(6)
A_SC_Q, A_SH_Q, A_SC_K, A_SH_K, A_SC_T, A_SH_T, A_G_A, A_G_T = range(8)


def _ext():
    from miniworld_engine.kernels.dit_variants import cuda

    return cuda.train_ext()


def _enabled(stack, single) -> bool:
    from miniworld_engine.modules.exceptions import ImplementationType

    cfg = settings.current()
    return (stack.implementation != ImplementationType.PYTORCH and single.is_cuda and single.ndim == 3 and single.dtype == torch.float32
            and not cfg.force_portable_kernels and getattr(cfg, "dit_variant_train_cuda", True)
            and torch.cuda.get_device_capability(single.device) == (9, 0))


def _no_grad_needed(stack, single, cond, pair) -> bool:
    return not (torch.is_grad_enabled() and any(t.requires_grad for t in (single, cond, pair, *stack.parameters())))


def serves(stack, single, cond, pair, mask) -> bool:
    """An fp32 token stack call without a gradient: 768 wide x 16 heads, L a multiple of 128, one conditioning row set, pair 128 / 256 wide."""
    from miniworld_engine.kernels.augmented_attention.cuda import tf32_inference_core_supported

    if not _enabled(stack, single) or not _no_grad_needed(stack, single, cond, pair):
        return False
    a, length, d = single.shape
    blk = stack.blocks[0]
    return (cond.ndim == 2 and cond.shape[0] == length and tuple(pair.shape[:2]) == (length, length) and pair.shape[-1] in (128, 256)
            and (mask is None or tuple(mask.shape) == (length,)) and blk.expand.weight.shape[0] == 4 * d and a > 0
            and tf32_inference_core_supported(torch.float32, length, d, blk.n_head, single.device.index or 0))


def atom_serves(stack, single, cond, pair, mask, uid) -> bool:
    """An fp32 atom stack call without a gradient: 128 wide, N a multiple of 64, conditioning ``[N, 128]``, trunked pair 16 wide."""
    if not _enabled(stack, single) or not _no_grad_needed(stack, single, cond, pair):
        return False
    a, n, d = single.shape
    return (d == 128 and n % 64 == 0 and cond.ndim == 2 and tuple(cond.shape) == (n, d) and tuple(pair.shape) == (n // 32, 32, 128, 16)
            and (mask is None or tuple(mask.shape) == (n,)) and (uid is None or tuple(uid.shape) == (n,))
            and stack.blocks[0].expand.weight.shape[0] == 4 * d and a > 0)


def _mod_entries(norm, d, dev):
    if norm.identity_scale:
        return ([norm.to_scale.weight.float(), norm.to_shift.weight.float()],
                [torch.ones(d, device=dev), torch.zeros(d, device=dev)])
    w = norm.cond_norm_weight.float()[None, :]
    return ([norm.to_scale.weight.float() * w, norm.to_shift.weight.float() * w],
            [norm.to_scale.bias.float(), torch.zeros(d, device=dev)])


# --------------------------------------------------------------------------------------------------------------------- token
def _token_pack(stack):
    blocks = list(stack.blocks)
    b0 = blocks[0]
    d = b0.to_q.weight.shape[0]
    dev = b0.to_q.weight.device
    qs = LOG2E / math.sqrt(d // b0.n_head)
    mod_w, mod_b, gate_w, gate_b, pair_w, per = [], [], [], [], [], []
    for blk in blocks:
        for norm in (blk.attn_adaln, blk.trans_adaln):
            w, b = _mod_entries(norm, d, dev)
            mod_w += w
            mod_b += b
        gate_w += [blk.attn_zero.weight.float(), blk.trans_zero.weight.float()]
        gate_b += [blk.attn_zero.bias.float(), blk.trans_zero.bias.float()]
        pair_w.append(blk.to_pair_bias.weight.float() * blk.pair_norm.weight.float()[None, :] * LOG2E)
        q_scale = 1.0 if blk.kq_norm else qs
        entry = {
            "wqkg": torch.cat([blk.to_q.weight.float() * q_scale, blk.to_k.weight.float(), *((blk.to_g.weight.float(),) if blk.gating else ())]).contiguous(),
            "bqkg": torch.cat([blk.to_q.bias.float() * q_scale, torch.zeros((2 if blk.gating else 1) * d, device=dev)]).contiguous(),
            "wv": blk.to_v.weight.float().contiguous(), "wo": blk.to_out.weight.float().contiguous(),
            "wexp": torch.cat([blk.expand.weight.float(), *((blk.up.weight.float(),) if blk.up_gate else ())]).contiguous(),
            "wsq": blk.squeeze.weight.float().contiguous(),
        }
        if blk.kq_norm:
            entry["qk"] = ((blk.q_norm.weight.float() * qs).contiguous(), (blk.q_norm.bias.float() * qs).contiguous(),
                           blk.k_norm.weight.float().contiguous(), blk.k_norm.bias.float().contiguous())
        per.append(entry)
    return {"mod_w": torch.cat(mod_w).contiguous(), "mod_b": torch.cat(mod_b).contiguous(), "gate_w": torch.cat(gate_w).contiguous(),
            "gate_b": torch.cat(gate_b).contiguous(), "pair_w": torch.cat(pair_w).contiguous(), "blocks": per}


def _token_bias(pack, pair, mask, eps, nbh):
    from miniworld_engine.kernels.dit_variants.cuda import train
    from miniworld_engine.kernels.dit_variants.cuda.train import _key_perm

    length, c = pair.shape[0], pair.shape[-1]
    zhat = train.pair_ln_fwd(pair.reshape(length * length, c).contiguous(), eps, False)
    bias = torch.mm(pack["pair_w"], zhat.t()).view(nbh, length, length)
    perm = _key_perm(length, pair.device)
    bias = bias[..., perm].contiguous()
    if mask is not None:
        bias.masked_fill_(~mask.bool()[perm][None, None, :], float("-inf"))
    return bias, pair, mask


def inference(stack, single, cond, pair, mask):
    """The fp32 token stack's forward, no gradient: x [A, L, D] -> [A, L, D]."""
    from miniworld_engine.kernels.augmented_attention import cuda as attn

    ext = _ext()
    blocks = list(stack.blocks)
    b0 = blocks[0]
    nb, heads = len(blocks), b0.n_head
    a, length, d = single.shape
    m = a * length
    dev = single.device
    pack = _bf16._static_pack(_PACKS, stack, _token_pack)
    sig = not b0.attn_adaln.identity_scale
    s = cond.float().contiguous()
    sm = F.layer_norm(s, s.shape[-1:], eps=1e-5) if sig else s
    t_mod = torch.addmm(pack["mod_b"], sm, pack["mod_w"].t())          # [L, nb 4 D]: per block scA | shA | scT | shT
    t_gate = torch.addmm(pack["gate_b"], s, pack["gate_w"].t())        # [L, nb 2 D]: per block gA | gT
    key = (pair.data_ptr(), pair._version, tuple(pair.shape), None if mask is None else (mask.data_ptr(), mask._version), pack["pair_w"].data_ptr())
    bias_all = _capture.lookup_inputs(_TOKEN_BIASES, key, lambda: _token_bias(pack, pair, mask, b0.pair_norm.eps, nb * heads), limit=2)[0]

    f32 = torch.float32
    st = torch.empty(m, 2, device=dev, dtype=f32)
    stq = torch.empty(m, 4, device=dev, dtype=f32) if any(b.kq_norm for b in blocks) else None
    xa = torch.empty(m, d, device=dev, dtype=f32)
    xt = torch.empty_like(xa) if stack.parallel else xa
    qkg = torch.empty(m, 3 * d, device=dev, dtype=f32)
    vt = torch.empty(d, m, device=dev, dtype=f32)
    width = b0.expand.weight.shape[0] // 2
    ab = torch.empty(m, (3 if b0.up_gate else 2) * width, device=dev, dtype=f32)
    h = torch.empty(m, width, device=dev, dtype=f32)
    y2 = torch.empty(m, d, device=dev, dtype=f32)
    xp = torch.empty(m, d, device=dev, dtype=f32)
    if not b0.gating:
        qkg[:, 2 * d:].fill_(_NO_GATE)
    x = single.reshape(m, d).contiguous()
    cols = lambda c: c * d  # noqa: E731
    for i, (blk, w) in enumerate(zip(blocks, pack["blocks"], strict=True)):
        tab = torch.cat([t_mod[:, 4 * d * i:4 * d * (i + 1)], t_gate[:, 2 * d * i:2 * d * (i + 1)]], dim=1)
        if stack.parallel:
            ext.ln_mod_fwd(x, tab, [cols(T_SC_A), cols(T_SC_T)], [cols(T_SH_A), cols(T_SH_T)], [xa, xt], st, length, b0.adaln_eps, sig)
        else:
            ext.ln_mod_fwd(x, tab, [cols(T_SC_A)], [cols(T_SH_A)], [xa], st, length, b0.adaln_eps, sig)
        torch.addmm(w["bqkg"], xa, w["wqkg"].t(), out=qkg[:, :(3 if blk.gating else 2) * d])
        torch.mm(w["wv"], xa.t(), out=vt)
        if blk.kq_norm:
            wq, bq, wk, bk = w["qk"]
            ext.qkv_prep(qkg[:, :d], qkg[:, d:2 * d], qkg[:, :d], qkg[:, :d], qkg[:, d:2 * d], None, stq, wq, bq, wk, bk, 1.0, blk.q_norm.eps, True, True)
        else:
            ext.qkv_prep(qkg[:, :d], qkg[:, d:2 * d], qkg[:, :d], qkg[:, :d], qkg[:, d:2 * d], None, None, None, None, None, None, 1.0, 1e-5, False, True)
        attn.tf32_gated_inference(qkg, vt, bias_all, i, a, 2 * d, heads, d // heads)
        ya = torch.mm(qkg[:, :d], w["wo"].t())
        out = torch.empty(m, d, device=dev, dtype=f32)
        if stack.parallel:
            torch.mm(xt, w["wexp"].t(), out=ab)
            ext.glu_fwd(ab, h, blk.up_gate)
            torch.mm(h, w["wsq"].t(), out=y2)
            ext.res_out_fwd(x, ya, tab, cols(T_G_A), y2, cols(T_G_T), out, length)
        else:
            ext.res_ln_fwd(x, ya, tab, cols(T_G_A), cols(T_SC_T), cols(T_SH_T), xp, xt, st, length, b0.adaln_eps, sig)
            torch.mm(xt, w["wexp"].t(), out=ab)
            ext.glu_fwd(ab, h, blk.up_gate)
            torch.mm(h, w["wsq"].t(), out=y2)
            ext.res_out_fwd(xp, y2, tab, cols(T_G_T), None, 0, out, length)
        x = out
    return x.view(a, length, d)


# ---------------------------------------------------------------------------------------------------------------------- atom
def _atom_pack(stack):
    blocks = list(stack.blocks)
    b0 = blocks[0]
    d = 128
    dev = b0.to_q.weight.device
    mod_w, mod_b, gate_w, gate_b, per = [], [], [], [], []
    for i, blk in enumerate(blocks):
        for norm in (blk.q_adaln, blk.k_adaln, blk.trans_adaln):
            w, b = _mod_entries(norm, d, dev)
            mod_w += w
            mod_b += b
        gate_w += [blk.attn_zero.weight.float(), blk.trans_zero.weight.float()]
        gate_b += [blk.attn_zero.bias.float(), blk.trans_zero.bias.float()]
        if stack.per_block_pair_norm:
            gamma, wb = blk.pair_norm.weight, blk.to_pair_bias.weight
        else:
            gamma, wb = stack.pair_norm.weight, stack.to_pair_bias.weight[4 * i:4 * i + 4]
        entry = {
            "wq": blk.to_q.weight.float().contiguous(), "bq": blk.to_q.bias.float().contiguous(), "wk": blk.to_k.weight.float().contiguous(),
            "wv": blk.to_v.weight.float().contiguous(), "wexp": torch.cat([blk.expand.weight.float(), *((blk.up.weight.float(),) if blk.up_gate else ())]).contiguous(),
            "wsq": blk.squeeze.weight.float().contiguous(), "gamma": gamma.float().contiguous(), "wb": wb.float().contiguous(),
        }
        if blk.gating:
            entry["wg"] = blk.to_g.weight.float().contiguous()
        if blk.project_output:
            entry["wo"] = blk.to_out.weight.float().contiguous()
        if blk.kq_norm:
            entry["qk"] = tuple(t.float().contiguous() for t in (blk.q_norm.weight, blk.q_norm.bias, blk.k_norm.weight, blk.k_norm.bias))
        per.append(entry)
    return {"mod_w": torch.cat(mod_w).contiguous(), "mod_b": torch.cat(mod_b).contiguous(), "gate_w": torch.cat(gate_w).contiguous(),
            "gate_b": torch.cat(gate_b).contiguous(), "blocks": per}


def atom_inference(stack, single, cond, pair, mask, uid):
    """The fp32 atom stack's forward, no gradient: x [A, N, 128] -> [A, N, 128]."""
    from miniworld_engine.kernels.augmented_attention.cuda import sm100_atom_local as local

    ext = _ext()
    blocks = list(stack.blocks)
    b0 = blocks[0]
    a, n, d = single.shape
    m = a * n
    dev = single.device
    f32 = torch.float32
    pack = _bf16._static_pack(_PACKS, stack, _atom_pack)
    sig = not b0.q_adaln.identity_scale
    c = cond.float().contiguous()
    cm = F.layer_norm(c, (d,), eps=1e-5) if sig else c
    t_mod = torch.addmm(pack["mod_b"], cm, pack["mod_w"].t())          # [N, nb 6 D]: per block qsc | qsh | ksc | ksh | tsc | tsh
    t_gate = torch.addmm(pack["gate_b"], c, pack["gate_w"].t())        # [N, nb 2 D]
    kmask = torch.ones(n, dtype=torch.bool, device=dev) if mask is None else mask.bool()
    slide = stack.key_window == "slide"
    eps = (b0.pair_norm if stack.per_block_pair_norm else stack.pair_norm).eps
    z = pair.float().contiguous()
    key = (pair.data_ptr(), pair._version, tuple(pair.shape), None if mask is None else (mask.data_ptr(), mask._version),
           None if uid is None else (uid.data_ptr(), uid._version), id(pack), stack.key_window)
    biases = _capture.lookup_inputs(_ATOM_BIASES, key, lambda: (_bf16._atom_biases(pack, z, kmask, uid, n, eps, slide), pair, mask, uid), limit=2)[0]

    st = torch.empty(m, 2, device=dev, dtype=f32)
    stq = torch.empty(m, 4, device=dev, dtype=f32) if any(b.kq_norm for b in blocks) else None
    xq, xk = torch.empty(m, d, device=dev, dtype=f32), torch.empty(m, d, device=dev, dtype=f32)
    xt = torch.empty_like(xq) if stack.parallel else xq
    k, v = torch.empty(m, d, device=dev, dtype=f32), torch.empty(m, d, device=dev, dtype=f32)
    g = torch.empty(m, d, device=dev, dtype=f32)
    width = b0.expand.weight.shape[0] // 2
    ab = torch.empty(m, (3 if b0.up_gate else 2) * width, device=dev, dtype=f32)
    h = torch.empty(m, width, device=dev, dtype=f32)
    y2 = torch.empty(m, d, device=dev, dtype=f32)
    xp = torch.empty(m, d, device=dev, dtype=f32)
    x = single.reshape(m, d).contiguous()
    cols = lambda cc: cc * d  # noqa: E731
    for i, (blk, w) in enumerate(zip(blocks, pack["blocks"], strict=True)):
        tab = torch.cat([t_mod[:, 6 * d * i:6 * d * (i + 1)], t_gate[:, 2 * d * i:2 * d * (i + 1)]], dim=1)
        outs, scs, shs = [xq, xk], [cols(A_SC_Q), cols(A_SC_K)], [cols(A_SH_Q), cols(A_SH_K)]
        if stack.parallel:
            outs, scs, shs = outs + [xt], scs + [cols(A_SC_T)], shs + [cols(A_SH_T)]
        ext.ln_mod_fwd(x, tab, scs, shs, outs, st, n, b0.adaln_eps, sig)
        q = torch.addmm(w["bq"], xq, w["wq"].t())
        torch.mm(xk, w["wk"].t(), out=k)
        torch.mm(xk, w["wv"].t(), out=v)
        if blk.gating:
            torch.mm(xq, w["wg"].t(), out=g)
        if blk.kq_norm:
            wq, bq, wk, bk = w["qk"]
            ext.qkv_prep(q, k, v, q, k, v, stq, wq, bq, wk, bk, 1.0, blk.q_norm.eps, True, True)
        else:
            ext.qkv_prep(q, k, v, q, k, v, None, None, None, None, None, 1.0, 1e-5, False, True)
        bias2, edge = biases[i]
        o, _ = local.attn_fwd_mma32(q.view(a, n, d), k.view(a, n, d), v.view(a, n, d), bias2)
        if slide:
            widx, keys, ebias = edge
            _bf16_free = _ext_edge()
            _bf16_free.edge_attn_fwd(q.view(a, n, d), k.view(a, n, d), v.view(a, n, d), keys, widx, ebias, o)
        o = o.view(m, d)
        if blk.gating:
            og = torch.empty_like(o)
            ext.gate_fwd(o, g, og)
        else:
            og = o
        ya = torch.mm(og, w["wo"].t()) if blk.project_output else og
        out = torch.empty(m, d, device=dev, dtype=f32)
        if stack.parallel:
            torch.mm(xt, w["wexp"].t(), out=ab)
            ext.glu_fwd(ab, h, blk.up_gate)
            torch.mm(h, w["wsq"].t(), out=y2)
            ext.res_out_fwd(x, ya, tab, cols(A_G_A), y2, cols(A_G_T), out, n)
        else:
            ext.res_ln_fwd(x, ya, tab, cols(A_G_A), cols(A_SC_T), cols(A_SH_T), xp, xt, st, n, b0.adaln_eps, sig)
            torch.mm(xt, w["wexp"].t(), out=ab)
            ext.glu_fwd(ab, h, blk.up_gate)
            torch.mm(h, w["wsq"].t(), out=y2)
            ext.res_out_fwd(xp, y2, tab, cols(A_G_T), None, 0, out, n)
        x = out
    return x.view(a, n, d)


def _ext_edge():
    from miniworld_engine.kernels.dit_variants import cuda

    return cuda.edge_ext()
