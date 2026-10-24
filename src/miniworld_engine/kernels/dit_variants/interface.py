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

from miniworld_engine import settings
from miniworld_engine.kernels import _capture

LOG2E = 1.4426950408889634
#: gate logits of a block without a gating query: sigmoid(1e4) == 1.0 in fp32
_NO_GATE = 1.0e4
_PACKS: dict = {}
_BIASES: dict = {}


def _static_pack(store: dict, stack, build):
    """The stack's folded bf16 weights, made once per (parameter pointer, version) and reused by every later call, a CUDA-graph capture
    and its replays included: these modules are inference only, so their weights do not change under a graph, and repacking in every
    replay (what ``_capture.lookup``'s capture scope does) cost more than the step's kernels (RoseTTAFold3, 24 blocks, L 384: 5.75 against
    2.59 ms). A first call inside a capture records the pack (its replays repack); an in-place weight change after a capture needs a new
    capture, as it would for any cached state."""
    key = (id(stack), *((p.data_ptr(), p._version) for p in stack.parameters()))
    hit = store.get(key)
    if hit is None:
        if len(store) >= 8:
            store.pop(next(iter(store)))
        hit = store[key] = build(stack)
    return hit


def serves(stack, single, cond, pair, mask) -> bool:
    """A bf16 call without a gradient on H100 at the AF3 token core's layout (768 wide, 16 x 48 heads, L a multiple of 128),
    the conditioning shared by the samples."""
    if torch.is_grad_enabled() or not single.is_cuda or single.dtype is not torch.bfloat16:
        return False
    if torch.cuda.get_device_capability(single.device) != (9, 0) or settings.current().force_triton_kernels:
        return False
    from miniworld_engine.kernels.augmented_attention.cuda import (
        inference_core_supported,
    )

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
    pack = _static_pack(_PACKS, stack, _pack)
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
        mods = [(msa, mba, xa), *([(mst, mbt, xt)] if stack.parallel else [])]
        if i == 0 or stack.parallel:   # a sequential block's LN + AdaLN is the previous block's last residual pass (below)
            cuda.ln_mod_rows(x, mods, L, eps, sig, xin=single.reshape(M, D) if i == 0 else None)
        torch.addmm(w["bqkv"], xa, w["wqkv"].t(), out=qkvg if blk.gating else qkvg[:, :3 * D])
        if blk.kq_norm:
            wq, bq, wk, bk, qeps = w["qk"]
            cuda.qk_ln_rows(qkvg[:, :D], qkvg[:, D:2 * D], wq, bq, wk, bk, qeps)
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
            if i + 1 < nb:
                o = 4 * (i + 1) * D
                cuda.res_rows(x, torch.mm(h, w["wsq"].t()), gt, L, ms=t_mod[:, o:o + D], mb=t_mod[:, o + D:o + 2 * D], out_ln=xa, eps=eps, sig=sig)
            else:
                cuda.res_rows(x, torch.mm(h, w["wsq"].t()), gt, L, final=last)
    return out.view(A, L, D)


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
    if torch.cuda.get_device_capability(single.device) != (9, 0) or settings.current().force_triton_kernels:
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


def _atom_biases(pack, z, kmask, uid, n, eps, slide):
    """Per block: the clipped windows' bias (``pair_bias_mma``, log2 units, -inf off the keys, Chai-1's other-conformer keys -inf) and,
    for the slid key window, its edge windows' (window index [5], key index [5, 128], bias [5, h, 32, 128] natural units with -inf off
    the keys): everything that depends on the pair / mask / uid only, made once per pair."""
    from miniworld_engine.kernels.augmented_attention.cuda import (
        sm100_atom_local as local,
    )

    nwin = n // 32
    same = None
    if uid is not None:
        # the clipped window's key of slot j: 32 w - 48 + j; keys outside [0, N) are -inf already. A padding query keeps its keys.
        idx = (32 * torch.arange(nwin, device=z.device) - 48)[:, None] + torch.arange(128, device=z.device)
        same = (uid.view(nwin, 32, 1) == uid[idx.clamp(0, n - 1)][:, None, :]) | ~kmask.view(nwin, 32, 1)
    edge = None
    if slide:
        # windows 0, 1 and the last three real ones: the only ones whose slid window is not the clipped one (chosen on the device)
        n_real = kmask.sum()
        last = (n_real - 1) // 32
        widx = torch.stack([last * 0, last * 0 + 1, last - 2, last - 1, last]).clamp(0, nwin - 1)
        starts = torch.minimum((32 * widx - 48).clamp(min=0), (n_real - 128).clamp(min=0))
        keys = starts[:, None] + torch.arange(128, device=z.device)
        valid = (keys < n_real)[:, None, :]                                             # [5, 1, 128]
        keys = keys.clamp(max=n - 1)
        if uid is not None:
            rows = 32 * widx[:, None] + torch.arange(32, device=z.device)
            valid = valid & ((uid[rows][:, :, None] == uid[keys][:, None, :]) | ~kmask[rows][:, :, None])
        zz = z[widx].float()
        edge = (widx, keys, valid, zz)
    out = []
    for w in pack["blocks"]:
        b = local.pair_bias_mma(z, w["gamma"], w["wb"], kmask, n, eps)
        if same is not None:
            b.masked_fill_(~same[None], float("-inf"))
        e = None
        if edge is not None:
            widx, keys, valid, zz = edge
            eb = F.linear(F.layer_norm(zz, zz.shape[-1:], w["gamma"], None, eps), w["wb"]).permute(0, 3, 1, 2)   # [5, h, 32, 128]
            e = (widx, keys, eb.masked_fill(~valid[:, None], float("-inf")).contiguous())
        out.append((b, e))
    return out


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
            cuda.gate_rows(o, torch.mm(xq, w["wg"].t()))
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
