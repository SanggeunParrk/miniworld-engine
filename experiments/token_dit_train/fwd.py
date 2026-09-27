"""Python front door for the forward kernel (prep in torch for now). TDT_FWD=1 (default): attn_fwd.cu, 2:
attn_fwd2.cu (resident bias, samples looped in the CTA; TDT_SCH samples per CTA, default picked for the wave count)."""
import math
import os
import torch
from . import ext
from .ref import LOG2E, D


def prep(q, k, v, bias, mask):
    A, L, HD = q.shape
    qs = (q * (LOG2E / math.sqrt(D))).to(torch.bfloat16).reshape(A * L, HD)
    kb = k.to(torch.bfloat16).reshape(A * L, HD)
    vb = v.to(torch.bfloat16).reshape(A * L, HD)
    bb = (bias * LOG2E).to(torch.bfloat16).contiguous()          # log2 units, as the kernels seed it
    km = None
    if mask is not None:
        km = torch.where(mask, 0.0, -1e30).float().contiguous()
    return qs, kb, vb, bb, km


def pick_sch(A, L, H=16, sms=132):
    """Samples per CTA: the largest divisor of A whose last wave is at least 90 % full (the bias costs 2 / sch bytes
    a pair, so more is better until the wave quantisation bites)."""
    ctas = (L // 384) * (L // 128) * H
    for s in range(A, 0, -1):
        if A % s == 0:
            n = ctas * (A // s)
            if n / (math.ceil(n / sms) * sms) >= 0.9:
                return s
    return 1


def attn_fwd(qs, kb, vb, bb, km, A, L):
    o = torch.empty(A * L, qs.shape[1], device=qs.device, dtype=torch.float32)
    lse = torch.empty(A, bb.shape[0], L, device=qs.device, dtype=torch.float32)
    if os.environ.get("TDT_FWD", "1") == "1":
        ext("attn_fwd").attn_fwd(qs, kb, vb, bb, km, o, lse)
    else:
        sch = int(os.environ.get("TDT_SCH", 0)) or pick_sch(A, L, bb.shape[0])
        ext("attn_fwd2").attn_fwd2(qs, kb, vb, bb, km, o, lse, sch)
    return o.view(A, L, -1), lse
