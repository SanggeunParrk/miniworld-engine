"""Python front door for the backward kernels (the prep in torch for now)."""
import torch
from . import ext


def bwd_prep(do, o, A, L, H=16):
    """dO bf16 [A*L, 768] and D = rowsum(dO * O) [A, H, L] (natural units)."""
    dd = (do.float() * o.float()).view(A, L, H, -1).sum(-1).transpose(1, 2).contiguous()
    return do.to(torch.bfloat16).reshape(A * L, -1).contiguous(), dd


def attn_dq(qs, kb, vb, dob, bb, km, lse, dd, A, L):
    dq = torch.empty(A * L, qs.shape[1], device=qs.device, dtype=torch.float32)
    extra = () if L % 192 == 0 else ("NWG=2",)                      # the default build tiles L by 192
    ext("attn_dq", extra).attn_dq(qs, kb, vb, dob, bb, km, lse, dd, dq)
    return dq.view(A, L, -1)


def attn_dkv(qs, kb, vb, dob, bb, km, lse, dd, A, L):
    """dK, dV [A, L, 768] fp32 and dbias [H, L, L] fp32 (natural units)."""
    H = bb.shape[0]
    dk = torch.empty(A * L, qs.shape[1], device=qs.device, dtype=torch.float32)
    dv = torch.empty_like(dk)
    dbt = torch.zeros(H, L, L, device=qs.device, dtype=torch.float32)
    ext("attn_dkv").attn_dkv(qs, kb, vb, dob, bb, km, lse, dd, dk, dv, dbt)
    return dk.view(A, L, -1), dv.view(A, L, -1), dbt.transpose(1, 2)


def pick_sch_dkv(A, L, H=16, sms=132):
    """Samples per dkv2 CTA: the largest divisor of A whose last wave is at least 90 % full."""
    import math
    ctas = (L // 192) * (L // 128) * H
    for s in range(A, 0, -1):
        if A % s == 0:
            n = ctas * (A // s)
            if n / (math.ceil(n / sms) * sms) >= 0.9:
                return s
    return 1


def attn_dkv2(qs, kb, vb, dob, bb, km, lse, dd, A, L, sch=None):
    """attn_dkv2.cu: dbias summed over the samples on chip (per chunk of sch samples; the chunks summed here)."""
    import os
    H = bb.shape[0]
    sch = sch or int(os.environ.get("TDT_DKV_SCH", 0)) or pick_sch_dkv(A, L, H)
    dk = torch.empty(A * L, qs.shape[1], device=qs.device, dtype=torch.float32)
    dv = torch.empty_like(dk)
    dbt = torch.empty(A // sch, H, L, L, device=qs.device, dtype=torch.float32)
    ext("attn_dkv2").attn_dkv2(qs, kb, vb, dob, bb, km, lse, dd, dk, dv, dbt, sch)
    return dk.view(A, L, -1), dv.view(A, L, -1), dbt.sum(0).transpose(1, 2)


def attn_dqb(qs, kb, vb, dob, bb, km, lse, dd, A, L):
    """attn_dqb.cu: dQ [A, L, 768] fp32 and dbias [H, L, L] fp32 (natural units), the samples' dS summed 3 at a time
    on chip before the L2 atomics."""
    H = bb.shape[0]
    dq = torch.empty(A * L, qs.shape[1], device=qs.device, dtype=torch.float32)
    db = torch.zeros(H, L, L, device=qs.device, dtype=torch.float32)
    ext("attn_dqb").attn_dqb(qs, kb, vb, dob, bb, km, lse, dd, dq, db)
    return dq.view(A, L, -1), db


def attn_dkv_nobias(qs, kb, vb, dob, bb, km, lse, dd, A, L):
    """attn_dkv.cu built without dbias (DBIAS=0): dK, dV only."""
    dk = torch.empty(A * L, qs.shape[1], device=qs.device, dtype=torch.float32)
    dv = torch.empty_like(dk)
    dbt = torch.empty(1, device=qs.device, dtype=torch.float32)
    ext("attn_dkv", ("DBIAS=0",)).attn_dkv(qs, kb, vb, dob, bb, km, lse, dd, dk, dv, dbt)
    return dk.view(A, L, -1), dv.view(A, L, -1)
