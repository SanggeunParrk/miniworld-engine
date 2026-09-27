"""Fused glue for the training core: one pass each instead of torch's scale / cast / multiply / sum chains."""
import torch
import triton
import triton.language as tl


@triton.jit
def _prep_qkv_kernel(q, k, v, qs, kb, vb, n, scale, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    tl.store(qs + offs, (tl.load(q + offs, mask=m) * scale).to(tl.bfloat16), mask=m)
    tl.store(kb + offs, tl.load(k + offs, mask=m).to(tl.bfloat16), mask=m)
    tl.store(vb + offs, tl.load(v + offs, mask=m).to(tl.bfloat16), mask=m)


def prep_qkv(q, k, v, scale):
    """fp32 q, k, v (any shape, contiguous) -> bf16 q * scale, k, v flattened to [-1, 768]."""
    q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
    n = q.numel()
    qs, kb, vb = (torch.empty(n, device=q.device, dtype=torch.bfloat16) for _ in range(3))
    BLOCK = 4096
    _prep_qkv_kernel[(triton.cdiv(n, BLOCK),)](q, k, v, qs, kb, vb, n, scale, BLOCK=BLOCK, num_warps=8)
    return qs.view(-1, 768), kb.view(-1, 768), vb.view(-1, 768)


@triton.jit
def _prep_do_kernel(do, o, dob, dd, L, H: tl.constexpr, DH: tl.constexpr, DP: tl.constexpr, ROWS: tl.constexpr):
    # ROWS tokens of one sample x all H heads; D[a, h, l] = sum_d dO * O
    pid = tl.program_id(0)
    r = pid * ROWS + tl.arange(0, ROWS)                                   # token rows (a * L + l)
    hh = tl.arange(0, H)
    dcol = tl.arange(0, DP)
    col = hh[:, None] * DH + dcol[None, :]                                # [H, DP]
    cm = dcol[None, :] < DH
    ptr = r[:, None, None] * (H * DH) + col[None, :, :]                   # [ROWS, H, DP]
    msk = cm[None, :, :] & (r[:, None, None] >= 0)
    g = tl.load(do + ptr, mask=msk, other=0.0)
    x = tl.load(o + ptr, mask=msk, other=0.0)
    tl.store(dob + ptr, g.to(tl.bfloat16), mask=msk)
    s = tl.sum(g * x, axis=2)                                             # [ROWS, H]
    a = r // L
    l = r % L
    tl.store(dd + (a[:, None] * H + hh[None, :]) * L + l[:, None], s)


def prep_do(do, o, A, L, H=16):
    """dO (fp32, [A*L, H*48] rows) -> bf16 dO and D = rowsum(dO * O) as [A, H, L] fp32."""
    do = do.reshape(A * L, H * 48).contiguous()
    o = o.reshape(A * L, H * 48)
    dob = torch.empty_like(do, dtype=torch.bfloat16)
    dd = torch.empty(A, H, L, device=do.device, dtype=torch.float32)
    ROWS = 8
    assert (A * L) % ROWS == 0
    _prep_do_kernel[((A * L) // ROWS,)](do, o, dob, dd, L, H=H, DH=48, DP=64, ROWS=ROWS, num_warps=8)
    return dob, dd


@triton.jit
def _bias_prep_kernel(b, out, n, scale, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    tl.store(out + offs, (tl.load(b + offs, mask=m) * scale).to(tl.bfloat16), mask=m)


def bias_prep(bias_hm, scale):
    """head-major fp32 bias [H, L, L] -> bf16 bias * scale (log2 units)."""
    bias_hm = bias_hm.contiguous()
    out = torch.empty_like(bias_hm, dtype=torch.bfloat16)
    n = bias_hm.numel()
    BLOCK = 4096
    _bias_prep_kernel[(triton.cdiv(n, BLOCK),)](bias_hm, out, n, scale, BLOCK=BLOCK, num_warps=8)
    return out
