"""Backward kernels against the fp64 autograd truth; the bf16-input floor for scale."""
import math
import sys
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from token_dit_train.ref import make, reference_grads, rel, LOG2E  # noqa: E402
from token_dit_train.fwd import prep, attn_fwd  # noqa: E402
from token_dit_train.bwd import bwd_prep, attn_dq, attn_dkv, attn_dkv2  # noqa: E402

ok = True
for L, A, bs, mf in ((384, 4, 1.0, 0.0), (768, 8, 1.0, 0.0), (384, 8, 4.0, 0.0), (384, 8, 1.0, 0.2)):
    q, k, v, bias, mask = make(A, L, bias_scale=bs, mask_frac=mf, seed=L + A)
    do = torch.randn_like(q)
    o_ref, dq_ref, dk_ref, dv_ref, db_ref = reference_grads(q, k, v, bias, do, mask)
    r = lambda t: t.bfloat16().float()  # noqa: E731
    bias_r = (bias * LOG2E).bfloat16().double() / LOG2E
    _, dq_b, dk_b, dv_b, db_b = reference_grads(r(q), r(k), r(v), bias_r, r(do), mask)
    args = prep(q, k, v, bias, mask)
    o, lse = attn_fwd(*args, A, L)
    dob, dd = bwd_prep(do, o, A, L)
    qs, kb, vb, bb, km = args
    dq = attn_dq(qs, kb, vb, dob, bb, km, lse, dd, A, L)
    dk, dv, db = attn_dkv(qs, kb, vb, dob, bb, km, lse, dd, A, L)
    dk2, dv2, db2 = attn_dkv2(qs, kb, vb, dob, bb, km, lse, dd, A, L)
    torch.cuda.synchronize()
    row = f"bwd: L{L} A{A} bias*{bs} mask{mf}:"
    for nm, got, ref, bref in (("dq", dq, dq_ref, dq_b), ("dk", dk, dk_ref, dk_b), ("dv", dv, dv_ref, dv_b),
                               ("dbias", db, db_ref, db_b), ("dk2", dk2, dk_ref, dk_b), ("dv2", dv2, dv_ref, dv_b),
                               ("dbias2", db2, db_ref, db_b)):
        e, eb = rel(got, ref), rel(bref, ref)
        good = e < 2 * eb + 2e-3 and math.isfinite(e)
        ok &= good
        row += f"  {nm} {e:.2e} ({eb:.2e}){'' if good else ' FAIL'}"
    print(row, flush=True)
print("bwd: ALL OK" if ok else "bwd: FAILED", flush=True)
